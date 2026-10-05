"""Precomputed demo scenarios and result tables for the static, shareable page."""

import base64
import json
import struct
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

from fidnn.demo.session import BATCH, DemoSession
from fidnn.inject.targets import BUCKETS
from fidnn.provenance import sidecar

BUDGETS = (1, 16)
DRAWS = 2


def png_data_uri(img: np.ndarray) -> str:
    """Minimal RGB PNG (see the PNG spec, §11.2): no image library needed for 32×32 thumbnails."""
    h, w, _ = img.shape
    raw = b"".join(b"\x00" + img[r].astype(np.uint8).tobytes() for r in range(h))

    def chunk(tag: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I",
                                                                      zlib.crc32(tag + data))
    png = (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
           + chunk(b"IDAT", zlib.compress(raw, 9)) + chunk(b"IEND", b""))
    return "data:image/png;base64," + base64.b64encode(png).decode()


def _rounded(a: np.ndarray, nd: int = 4) -> list:
    return [None if not np.isfinite(v) else round(float(v), nd) for v in np.ravel(a)]


def scenarios(session: DemoSession, gallery: np.ndarray, seed: int = 0) -> list[dict]:
    """Bucket × stratum × budget × draw, each evaluated on the gallery plus fixed filler probes."""
    filler = np.random.default_rng(seed).permutation(len(session.y))
    batch = np.concatenate([gallery, filler[~np.isin(filler, gallery)][:BATCH - len(gallery)]])
    rng = np.random.default_rng(seed)
    out = []
    for bucket in BUCKETS:
        for stratum in session.strata():
            for budget in BUDGETS:
                for draw in range(DRAWS):
                    session.reset()
                    session.add_flips(bucket, stratum, budget, rng)
                    ev = session.evaluate(batch)
                    n = len(gallery)
                    out.append({
                        "bucket": bucket, "stratum": stratum, "budget": budget, "draw": draw,
                        "flips": [{"layer": f["layer"], "index": int(f["flat_index"]),
                                   "bit": int(f["bit"]), **session.describe_flip(f)}
                                  for f in session.flips],
                        "fault_pred": [int(p) if np.isfinite(r).all() else -1
                                       for p, r in zip(np.nan_to_num(ev.fault_logits[:n]).argmax(1),
                                                       ev.fault_logits[:n])],
                        "label": ev.labels[:n].tolist(), "track": ev.tracks[:n].tolist(),
                        "d1": [_rounded(r) for r in ev.d1[:n]],
                        "first_alarm": ev.first_alarm[:n].tolist(), "d2": _rounded(ev.d2[:n]),
                        "msp": _rounded(ev.output_only["msp"][:n]),
                        "batch_accuracy": round(float(
                            (np.nan_to_num(ev.fault_logits).argmax(1) == ev.y).mean()), 4),
                    })
    session.reset()
    return out


def gallery(session: DemoSession, n: int = 24) -> np.ndarray:
    """Round-robin over classes, in probe order, so every class shows up."""
    by_class = [np.flatnonzero(session.y == c) for c in range(len(session.classes))]
    picks = [ix[k] for k in range(n) for ix in by_class if k < len(ix)]
    return np.array(picks[:n])


def results_tables(results_dir: Path, m0_dir: Path, model_id: str) -> dict:
    heads = sorted(results_dir.glob(f"{model_id}_*_headline.parquet"))
    head = pd.concat([pd.read_parquet(p) for p in heads], ignore_index=True) if heads else None
    inf_path = m0_dir / "inference.parquet"
    inf = pd.read_parquet(inf_path) if inf_path.exists() else None
    if inf is not None:
        inf = inf[(inf.model == model_id) & (inf.device == "cpu")][
            ["precision", "batch", "hooks", "tap_set", "n_taps", "median_ms"]]
    seeds = int(head.seed.nunique()) if head is not None else 0
    return {"headline": None if head is None else json.loads(head.to_json(orient="records")),
            "overhead_m0": None if inf is None else json.loads(inf.to_json(orient="records")),
            "seeds": seeds, "enough_seeds": seeds >= 5}


def export(session: DemoSession, out: Path, seed: int = 0, results_dir: Path | None = None,
           m0_dir: Path | None = None) -> dict:
    g = gallery(session)
    session.reset()
    ev = session.evaluate(g)  # no flips: the clean reference per image
    n = len(g)
    payload = {
        "model": session.model_id, "classes": list(session.classes), "taps": session.tap_ids,
        "tap_thresholds": _rounded(ev.d1_thresholds), "taus": session.taus,
        "images": [{"src": png_data_uri(session.images[i]), "y": int(session.y[i]),
                    "clean_pred": int(ev.clean_logits[k].argmax()),
                    "d1_clean": _rounded(ev.d1_clean[k])} for k, i in enumerate(g)],
        "scenarios": scenarios(session, g, seed),
        "results": (results_tables(results_dir, m0_dir, session.model_id)
                    if results_dir is not None else None),
        "provenance": sidecar({"budgets": BUDGETS, "draws": DRAWS, "images": n}, seed=seed,
                              device="cpu") | {"model_checksum": session.clean_checksum},
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, default=str))
    return payload
