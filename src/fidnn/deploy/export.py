"""`fidnn export`: a self-contained ONNX bundle of classifier + D2 monitor (deploy/README.md)."""

import hashlib
import json
import shutil
import time
import zipfile
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

from fidnn.data.cifar10 import CLASSES
from fidnn.data.prepare import load_arrays, model_classes, subset_classes
from fidnn.deploy.graph import MonitoredClassifier
from fidnn.detect.features import CleanFeatures, from_frame
from fidnn.models.checkpoint import load, manifest_entry
from fidnn.models.registry import MODELS, config_path
from fidnn.provenance import sidecar
from fidnn.taps.registry import taps

BUNDLE_FILES = Path(__file__).parent / "bundle"
CHECK_N = 16
SCORE_TOL = 0.1  # FP32 kernel rounding between PyTorch and ORT; the decision is checked exactly


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _latency_ms(session, images: np.ndarray, reps: int) -> float:
    session.run(None, {"images": images})
    t0 = time.perf_counter()
    for _ in range(reps):
        session.run(None, {"images": images})
    return (time.perf_counter() - t0) / reps * 1e3


def _overhead(model: torch.nn.Module, mean, std, mon_path: Path, x: np.ndarray, work: Path,
              reps: int = 200) -> dict:
    """C5: single-image CPU latency with and without the monitor, both under ORT."""
    import onnxruntime as ort

    class Plain(torch.nn.Module):
        def __init__(self, m, mean, std):
            super().__init__()
            self.m, self.mean, self.std = m, mean, std

        def forward(self, images):
            return self.m((images.permute(0, 3, 1, 2).float() / 255 - self.mean) / self.std)

    mon = ort.InferenceSession(str(mon_path), providers=["CPUExecutionProvider"])
    plain_path = work / "classifier_only.onnx"
    norm = Plain(model, *(torch.tensor(v).view(1, 3, 1, 1) for v in (mean, std)))
    torch.onnx.export(norm, (torch.from_numpy(x[:1]),), plain_path, input_names=["images"],
                      dynamo=True, external_data=False, verbose=False)
    plain = ort.InferenceSession(str(plain_path), providers=["CPUExecutionProvider"])
    one = x[:1]
    base, full = _latency_ms(plain, one, reps), _latency_ms(mon, one, reps)
    return {"device": "cpu", "runtime": f"onnxruntime {ort.__version__}", "batch": 1,
            "reps": reps, "classifier_ms": round(base, 4), "monitored_ms": round(full, 4),
            "added_latency_pct": round(100 * (full - base) / base, 1),
            "classifier_onnx_kb": round(plain_path.stat().st_size / 1024, 1),
            "measured_on": "export host — re-measure on the target device"}


def run(model_id: str, precision: str, tap_set: str, seed: int, detector: str, data_cfg: Path,
        data_dir: Path, models_dir: Path, features_dir: Path, detectors_dir: Path, out_dir: Path,
        log=print) -> dict:
    import onnxruntime as ort

    if precision != "fp32":
        raise SystemExit("export supports fp32 only: FX INT8 modules have no faithful ONNX "
                         "translation, so the monitor would no longer match its calibration")
    if detector != "D2":
        raise SystemExit("export supports D2 (one fused SVDD) only")
    stem = f"{model_id}_{precision}_{tap_set}_seed{seed}"
    bundle = joblib.load(detectors_dir / f"{stem}.joblib")
    d2 = bundle["detectors"]["D2"]
    cal = pd.read_parquet(detectors_dir / f"{stem}_calibration.parquet")
    cal = cal[cal.detector == detector].sort_values("alpha", ascending=False)
    extract = json.loads((features_dir / f"{stem}.json").read_text())
    sat = extract["saturation"]["thresholds"]
    tap_list = taps(model_id, tap_set)
    arrays = load_arrays(data_cfg, data_dir)
    subset = model_classes(config_path(model_id))
    classes = [CLASSES[c] for c in subset] if subset else list(CLASSES)

    model = load(model_id, seed, precision, models_dir)
    graph = MonitoredClassifier(load(model_id, seed, precision, models_dir),
                                [t.path for t in tap_list], [sat[t.tap_id] for t in tap_list],
                                arrays.mean, arrays.std, d2.normaliser, d2.model,
                                cal.tau.tolist()).eval()

    out = out_dir / stem
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    x, y = subset_classes(*arrays.of("clean_test"), subset)
    with torch.no_grad():  # before export: the traced module no longer feeds its eager recorders
        t = graph(torch.from_numpy(x[:CHECK_N]))
    np.savez_compressed(out / "check.npz", images=x[:CHECK_N], pred=t[2].numpy(),
                        score=t[3].numpy(), alarms=t[4].numpy())
    onnx_path = out / "model.onnx"
    torch.onnx.export(graph, (torch.from_numpy(x[:4]),), onnx_path, input_names=["images"],
                      output_names=["logits", "probs", "pred", "score", "alarms"],
                      dynamic_shapes={"images": {0: torch.export.Dim("batch")}},
                      dynamo=True, external_data=False, verbose=False)

    # parity: ORT against the fidnn pipeline on the cached clean_test features
    df = pd.read_parquet(features_dir / f"{stem}_clean.parquet")
    feats, idx = from_frame(df[df.split == "clean_test"], [t.tap_id for t in tap_list],
                            CleanFeatures)
    ref = np.empty(len(idx))
    ref[np.asarray(idx)] = d2.scores(feats)
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    o = dict(zip(["logits", "probs", "pred", "score", "alarms"], sess.run(None, {"images": x})))
    agree = {str(a): float((o["alarms"][:, k] == (ref > t)).mean())
             for k, (a, t) in enumerate(zip(cal.alpha, cal.tau))}
    score_err = float(np.abs(o["score"] - ref).max())
    if min(agree.values()) < 1.0 or score_err > SCORE_TOL:
        raise RuntimeError(f"ONNX monitor diverges from fidnn: alarm agreement {agree}, "
                           f"max score error {score_err:.4g}")
    log(f"{stem}: ONNX parity on {len(x)} clean_test images — alarms identical, "
        f"max score error {score_err:.2g}")

    for f in ("run.py", "README.md", "requirements.txt"):
        shutil.copy(BUNDLE_FILES / f, out / f)

    acc = manifest_entry(model_id, seed, models_dir)
    record = {
        "name": f"fidnn {MODELS[model_id].name} + {detector} fault monitor",
        "thesis_result": False,
        "files": {"model": "model.onnx", "check": "check.npz", "runner": "run.py"},
        "sha256": {"model": _sha256(onnx_path), "check": _sha256(out / "check.npz")},
        "input": {"name": "images", "dtype": "uint8", "shape": ["N", 32, 32, 3],
                  "layout": "NHWC RGB, 0–255; normalisation is inside the graph"},
        "outputs": {"logits": "(N, C) float32", "probs": "(N, C) softmax",
                    "pred": "(N,) int64 class index", "score": "(N,) float64 D2 score",
                    "alarms": "(N, len(alphas)) bool, one column per alpha"},
        "classes": classes,
        "classifier": {"model": model_id, "architecture": MODELS[model_id].name,
                       "precision": precision, "seed": seed,
                       "test_acc_fp32": acc.get("test_acc_fp32"),
                       "clean_test_acc_onnx": round(float((o["pred"] == y).mean()) * 100, 2),
                       "checkpoint_sha256": acc.get("sha256")},
        "monitor": {"detector": detector, "taps": [t.tap_id for t in tap_list],
                    "features_per_tap": 29, "alphas": cal.alpha.tolist(),
                    "thresholds": cal.tau.tolist(), "calibrated_on": "clean_cal",
                    "clean_test_fpr": {str(a): float(o["alarms"][:, k].mean())
                                       for k, a in enumerate(cal.alpha)},
                    "svdd": {"nu": d2.selection.nu, "gamma": float(d2.model._gamma),
                             "support_vectors": int(d2.model.support_vectors_.shape[0])}},
        "check": {"n": CHECK_N, "score_tolerance": SCORE_TOL,
                  "onnx_vs_fidnn": {"n": len(x), "alarm_agreement": agree,
                                    "max_score_error": score_err}},
        "overhead": _overhead(model, arrays.mean, arrays.std, onnx_path, x, out_dir),
        "caveats": [
            "Exported from one training seed: a demonstration, not a reported thesis number (C4).",
            ("Expects CIFAR-10-like 32×32 natural images; other inputs are out of distribution "
             "and may be flagged (or not) for reasons unrelated to faults."),
            "The clean false-alarm rate holds only for inputs like the calibration data.",
        ],
        **sidecar({"model": model_id, "precision": precision, "tap_set": tap_set,
                   "detector": detector}, seed=seed, device="cpu"),
    }
    (out_dir / "classifier_only.onnx").unlink(missing_ok=True)
    (out / "manifest.json").write_text(json.dumps(record, indent=2, default=str))
    archive = out_dir / f"{stem}.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as z:
        for f in sorted(out.iterdir()):
            z.write(f, f"{stem}/{f.name}")
    log(f"{archive}: {archive.stat().st_size / 1024:.0f} KB, "
        f"+{record['overhead']['added_latency_pct']}% latency on this CPU")
    return record
