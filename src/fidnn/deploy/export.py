"""`fidnn export`: a portable bundle — monitored ONNX, classifier formats, browser UI (README)."""

import base64
import hashlib
import json
import shutil
import urllib.request
import zipfile
from pathlib import Path

import joblib
import numpy as np
import onnx
import pandas as pd
import torch
from onnx import numpy_helper

from fidnn.data.cifar10 import CLASSES
from fidnn.data.prepare import load_arrays, model_classes, subset_classes
from fidnn.deploy import formats
from fidnn.deploy.graph import MonitoredClassifier
from fidnn.detect.features import CleanFeatures, from_frame
from fidnn.models.checkpoint import load, manifest_entry
from fidnn.models.registry import MODELS, config_path
from fidnn.provenance import sidecar
from fidnn.taps.registry import taps

BUNDLE_FILES = Path(__file__).parent / "bundle"
OUTPUTS = ["logits", "probs", "pred", "score", "alarms"]
CHECK_N = 16
UI_PER_CLASS = 20  # balanced, so per-class precision/recall in the UI rest on equal counts
SCORE_TOL = 0.1  # FP32 kernel rounding between PyTorch and ORT; the decision is checked exactly
ORT_WEB = "1.30.0"
ORT_WEB_FILES = ("ort.wasm.min.js", "ort-wasm-simd-threaded.mjs", "ort-wasm-simd-threaded.wasm")
CALIB_N = 512  # same clean_fit subsample size as `fidnn quantize`


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _ort_web(cache: Path, dest: Path) -> dict:
    """ONNX Runtime Web, pinned and cached, so the UI runs offline."""
    cache.mkdir(parents=True, exist_ok=True)
    dest.mkdir(parents=True, exist_ok=True)
    hashes = {}
    for name in ORT_WEB_FILES:
        f = cache / name
        if not f.exists():
            url = f"https://cdn.jsdelivr.net/npm/onnxruntime-web@{ORT_WEB}/dist/{name}"
            with urllib.request.urlopen(url, timeout=120) as r:
                f.write_bytes(r.read())
        shutil.copy(f, dest / name)
        hashes[name] = _sha256(f)
    return hashes


def _weight_table(path: Path) -> list[dict]:
    """Byte offset of each classifier weight's raw data inside the file, for in-browser flips."""
    raw = path.read_bytes()
    table = []
    for init in onnx.load(path).graph.initializer:
        w = numpy_helper.to_array(init)
        if not init.name.startswith("model.") or w.dtype != np.float32 or w.ndim < 2:
            continue
        data = w.astype("<f4").tobytes()
        offset = raw.find(data)
        if offset < 0 or raw.find(data, offset + 1) >= 0:
            continue  # not stored contiguously, or ambiguous: leave it out rather than guess
        table.append({"name": init.name.removeprefix("model."), "offset": offset,
                      "numel": int(w.size), "shape": list(w.shape)})
    return table


def _balanced(y: np.ndarray, per_class: int) -> np.ndarray:
    """First `per_class` indices of each class, interleaved so any prefix is near-balanced."""
    groups = [np.flatnonzero(y == c)[:per_class] for c in np.unique(y)]
    return np.stack(groups, 1).reshape(-1)


def _classifier_metrics(pred: np.ndarray, y: np.ndarray, classes: list[str]) -> dict:
    """Accuracy and per-class / macro precision, recall, F1 — reference for the UI."""
    from sklearn.metrics import precision_recall_fscore_support

    p, r, f, n = precision_recall_fscore_support(y, pred, labels=range(len(classes)),
                                                 zero_division=0)
    return {"n": len(y), "accuracy": float((pred == y).mean()),
            "macro": {"precision": float(p.mean()), "recall": float(r.mean()),
                      "f1": float(f.mean())},
            "per_class": {c: {"precision": float(p[i]), "recall": float(r[i]),
                              "f1": float(f[i]), "support": int(n[i])}
                          for i, c in enumerate(classes)}}


def _samples(x: np.ndarray, y: np.ndarray, classes: list[str]) -> list[dict]:
    return [{"pixels": base64.b64encode(np.ascontiguousarray(img).tobytes()).decode(),
             "label": classes[int(t)]} for img, t in zip(x, y)]


def run(model_id: str, precision: str, tap_set: str, seed: int, detector: str, data_cfg: Path,
        data_dir: Path, models_dir: Path, features_dir: Path, detectors_dir: Path, out_dir: Path,
        tflite: bool = True, cache_dir: Path = Path("artifacts/cache"), log=print) -> dict:
    if precision != "fp32":
        raise SystemExit("export starts from fp32: FX INT8 modules have no faithful ONNX "
                         "translation. The bundle's INT8 classifiers are quantised from it.")
    if detector != "D2":
        raise SystemExit("export supports D2 (one fused SVDD) only")
    stem = f"{model_id}_{precision}_{tap_set}_seed{seed}"
    d2 = joblib.load(detectors_dir / f"{stem}.joblib")["detectors"]["D2"]
    cal = pd.read_parquet(detectors_dir / f"{stem}_calibration.parquet")
    cal = cal[cal.detector == detector].sort_values("alpha", ascending=False)
    sat = json.loads((features_dir / f"{stem}.json").read_text())["saturation"]["thresholds"]
    tap_list = taps(model_id, tap_set)
    arrays = load_arrays(data_cfg, data_dir)
    subset = model_classes(config_path(model_id))
    classes = [CLASSES[c] for c in subset] if subset else list(CLASSES)
    x, y = subset_classes(*arrays.of("clean_test"), subset)
    fit_x, _ = subset_classes(*arrays.of("clean_fit"), subset)

    def fresh():
        return load(model_id, seed, precision, models_dir)

    graph = MonitoredClassifier(fresh(), [t.path for t in tap_list],
                                [sat[t.tap_id] for t in tap_list], arrays.mean, arrays.std,
                                d2.normaliser, d2.model, cal.tau.tolist()).eval()
    out = out_dir / stem
    if out.exists():
        shutil.rmtree(out)
    (out / "formats").mkdir(parents=True)

    with torch.no_grad():  # before export: the traced module no longer feeds its eager recorders
        t = graph(torch.from_numpy(x[:CHECK_N]))
    np.savez_compressed(out / "check.npz", images=x[:CHECK_N], pred=t[2].numpy(),
                        score=t[3].numpy(), alarms=t[4].numpy())
    model_path = formats.export_onnx(graph, torch.from_numpy(x[:4]), out / "model.onnx", OUTPUTS)

    # parity gate: ORT against the fidnn pipeline on the cached clean_test features
    import onnxruntime as ort

    df = pd.read_parquet(features_dir / f"{stem}_clean.parquet")
    feats, idx = from_frame(df[df.split == "clean_test"], [t.tap_id for t in tap_list],
                            CleanFeatures)
    ref = np.empty(len(idx))
    ref[np.asarray(idx)] = d2.scores(feats)
    sess = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    o = dict(zip(OUTPUTS, sess.run(None, {"images": x})))
    agree = {str(a): float((o["alarms"][:, k] == (ref > tau)).mean())
             for k, (a, tau) in enumerate(zip(cal.alpha, cal.tau))}
    score_err = float(np.abs(o["score"] - ref).max())
    if min(agree.values()) < 1.0 or score_err > SCORE_TOL:
        raise RuntimeError(f"ONNX monitor diverges from fidnn: alarm agreement {agree}, "
                           f"max score error {score_err:.4g}")
    log(f"{stem}: ONNX parity on {len(x)} clean_test images — alarms identical, "
        f"max score error {score_err:.2g}")

    # classifier-only formats, each evaluated on the same clean_test images
    fmt = out / "formats"
    c32 = formats.export_onnx(formats.Classifier(fresh(), arrays.mean, arrays.std),
                              torch.from_numpy(x[:4]), fmt / "classifier_fp32.onnx", ["logits"])
    c8 = formats.quantize_int8(c32, fmt / "classifier_int8.onnx", fit_x[:CALIB_N])
    entries = {
        "model.onnx": {"what": "classifier + fault monitor", "runtime": "ONNX Runtime",
                       **formats.evaluate(model_path, x, y, output=2)},
        "formats/classifier_fp32.onnx": {"what": "classifier only, FP32",
                                         "runtime": "ONNX Runtime, OpenCV DNN, TensorRT",
                                         **formats.evaluate(c32, x, y)},
        "formats/classifier_int8.onnx": {"what": "classifier only, static INT8 (QDQ)",
                                         "runtime": "ONNX Runtime",
                                         **formats.evaluate(c8, x, y)},
    }
    for src, key in [(model_path, "formats/model.ort"), (c8, "formats/classifier_int8.ort")]:
        ort_path = formats.to_ort(src)
        if ort_path.parent != fmt:
            ort_path = Path(shutil.move(ort_path, fmt / ort_path.name))
        src.with_suffix(".required_operators.config").replace(
            fmt / f"{ort_path.stem}.required_operators.config")
        entries[key] = {"what": f"{entries[str(src.relative_to(out))]['what']}, ORT format",
                        "runtime": "ONNX Runtime (incl. minimal builds, mobile)",
                        **formats.evaluate(ort_path, x, y, output=2 if src == model_path else 0)}
    if tflite:
        log(f"{stem}: converting to TFLite in an isolated environment …")
        nchw = formats.export_onnx(formats.Classifier(fresh(), arrays.mean, arrays.std, False),
                                   torch.zeros(1, 3, 32, 32), out / "_nchw.onnx", ["logits"],
                                   dynamic_batch=False)
        results = formats.tflite(nchw, fit_x[:256], x, y, fmt)
        nchw.unlink()
        what = {"classifier_fp32.tflite": ("FP32", "LiteRT / TFLite (Android, Linux, Pi)"),
                "classifier_fp16.tflite": ("FP16 weights", "LiteRT GPU / NPU delegates"),
                "classifier_int8.tflite": ("full INT8", "TFLite Micro, ESP32, STM32, Arduino")}
        for name, r in results.items():
            entries[f"formats/{name}"] = {"what": f"classifier only, {what[name][0]}",
                                          "runtime": what[name][1], **r}
        entries["formats/classifier_int8_tflite.h"] = {
            "what": "classifier_int8.tflite as a C array for MCU firmware",
            "runtime": "TFLite Micro"}
    for rel, e in entries.items():
        f = out / rel
        e.update({"kb": round(f.stat().st_size / 1024, 1), "sha256": _sha256(f)})
        if "accuracy" in e:
            e["accuracy"] = round(e["accuracy"] * 100, 2)

    # browser UI: runs model.onnx with ONNX Runtime Web, no Python needed beyond a static server
    ui = out / "ui"
    shutil.copytree(BUNDLE_FILES / "ui", ui)
    ort_web = _ort_web(cache_dir / f"onnxruntime-web-{ORT_WEB}", ui)
    pick = _balanced(y, UI_PER_CLASS)
    sx, sy = x[pick], y[pick]
    (ui / "samples.json").write_text(json.dumps(_samples(sx, sy, classes)))
    for f in ("run.py", "README.md", "requirements.txt"):
        shutil.copy(BUNDLE_FILES / f, out / f)

    base = entries["formats/classifier_fp32.onnx"]["latency_ms"]
    full = entries["model.onnx"]["latency_ms"]
    acc = manifest_entry(model_id, seed, models_dir)
    record = {
        "name": f"fidnn {MODELS[model_id].name} + {detector} fault monitor",
        "thesis_result": False,
        "files": {"model": "model.onnx", "check": "check.npz", "runner": "run.py", "ui": "ui/"},
        "sha256": {"model": _sha256(model_path), "check": _sha256(out / "check.npz")},
        "input": {"name": "images", "dtype": "uint8", "shape": ["N", 32, 32, 3],
                  "layout": "NHWC RGB, 0–255; normalisation is inside the graph"},
        "outputs": {"logits": "(N, C) float32", "probs": "(N, C) softmax",
                    "pred": "(N,) int64 class index", "score": "(N,) float64 D2 score",
                    "alarms": "(N, len(alphas)) bool, one column per alpha"},
        "classes": classes,
        "formats": entries,
        "classifier": {"model": model_id, "architecture": MODELS[model_id].name, "seed": seed,
                       "test_acc_fp32_canonical": acc.get("test_acc_fp32"),
                       "checkpoint_sha256": acc.get("sha256"),
                       "accuracy_measured_on": f"clean_test ({len(x)} images)"},
        "monitor": {"detector": detector, "taps": [t.tap_id for t in tap_list],
                    "features_per_tap": 29, "alphas": cal.alpha.tolist(),
                    "thresholds": cal.tau.tolist(), "calibrated_on": "clean_cal",
                    "clean_test_fpr": {str(a): float(o["alarms"][:, k].mean())
                                       for k, a in enumerate(cal.alpha)},
                    "svdd": {"nu": d2.selection.nu, "gamma": float(d2.model._gamma),
                             "support_vectors": int(d2.model.support_vectors_.shape[0])},
                    "only_in": "model.onnx and formats/model.ort"},
        "weights": _weight_table(model_path),
        "check": {"n": CHECK_N, "score_tolerance": SCORE_TOL,
                  "onnx_vs_fidnn": {"n": len(x), "alarm_agreement": agree,
                                    "max_score_error": score_err}},
        "overhead": {"device": "cpu", "runtime": f"onnxruntime {ort.__version__}", "batch": 1,
                     "classifier_ms": base, "monitored_ms": full,
                     "added_latency_pct": round(100 * (full - base) / base, 1),
                     "measured_on": "export host — re-measure on the target device"},
        "ui": {"runtime": f"onnxruntime-web {ORT_WEB}", "sha256": ort_web,
               "samples": f"{len(sx)} clean_test images, {UI_PER_CLASS} per class"},
        "metrics": {
            "measured_on": f"clean_test ({len(x)} images), original model",
            "classifier": _classifier_metrics(o["pred"], y, classes),
            "monitor_false_alarm_rate": {str(a): float(o["alarms"][:, k].mean())
                                         for k, a in enumerate(cal.alpha)},
            "note": ("Detection recall needs faulty inferences; the UI measures it live by "
                     "corrupting the model in the browser."),
        },
        "caveats": [
            "Exported from one training seed: a demonstration, not a reported thesis number (C4).",
            ("Expects CIFAR-10-like 32×32 natural images; other inputs are out of distribution "
             "and may be flagged (or not) for reasons unrelated to faults."),
            "The clean false-alarm rate holds only for inputs like the calibration data.",
            "Only model.onnx / model.ort carry the monitor; formats/classifier_* do not.",
        ],
        **sidecar({"model": model_id, "precision": precision, "tap_set": tap_set,
                   "detector": detector, "tflite": tflite}, seed=seed, device="cpu"),
    }
    (out / "manifest.json").write_text(json.dumps(record, indent=2, default=str))
    archive = out_dir / f"{stem}.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as z:
        for f in sorted(out.rglob("*")):
            if f.is_file():
                z.write(f, f"{stem}/{f.relative_to(out)}")
    log(f"{archive}: {archive.stat().st_size / 1024:.0f} KB, {len(entries)} files in "
        f"formats, monitor +{record['overhead']['added_latency_pct']}% latency on this CPU")
    return record
