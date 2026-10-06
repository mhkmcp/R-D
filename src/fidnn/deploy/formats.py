"""Classifier-only portable formats: ONNX FP32/INT8, ORT, TFLite (deploy/README.md)."""

import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

TFLITE_ENV = ("onnx2tf==1.28.8", "tensorflow==2.21.0", "tf_keras", "onnx", "onnxruntime",
              "onnx_graphsurgeon", "simple_onnx_processing_tools", "sng4onnx", "psutil",
              "ai_edge_litert", "onnxsim")


class Classifier(nn.Module):
    """Pixel values 0–255 in, logits out: the train-split normalisation lives in the graph.

    `nhwc_uint8` is the ONNX convention of the bundle; the NCHW float variant is what onnx2tf
    turns into a native NHWC TFLite input.
    """

    def __init__(self, model: nn.Module, mean, std, nhwc_uint8: bool = True):
        super().__init__()
        self.model, self.nhwc_uint8 = model.eval(), nhwc_uint8
        self.register_buffer("mean", torch.tensor(mean, dtype=torch.float32).view(1, 3, 1, 1) * 255)
        self.register_buffer("std", torch.tensor(std, dtype=torch.float32).view(1, 3, 1, 1) * 255)

    def forward(self, images):
        x = images.permute(0, 3, 1, 2).float() if self.nhwc_uint8 else images
        return self.model((x - self.mean) / self.std)


def export_onnx(module: nn.Module, example: torch.Tensor, path: Path, outputs: list[str],
                dynamic_batch: bool = True) -> Path:
    """See `torch.onnx.export`. A static batch-1 graph is what TFLite and MCU toolchains need."""
    shapes = {"images": {0: torch.export.Dim("batch")}} if dynamic_batch else None
    torch.onnx.export(module, (example,), path, input_names=["images"], output_names=outputs,
                      dynamic_shapes=shapes, dynamo=True, external_data=False, verbose=False)
    return path


def quantize_int8(fp32: Path, out: Path, calib: np.ndarray) -> Path:
    """Static QDQ INT8 (`onnxruntime.quantization.quantize_static`), calibrated on clean_fit."""
    from onnxruntime.quantization import (
        CalibrationDataReader,
        QuantFormat,
        QuantType,
        quantize_static,
    )
    from onnxruntime.quantization.shape_inference import quant_pre_process

    class Reader(CalibrationDataReader):
        def __init__(self):
            self.it = iter([{"images": calib[i:i + 32]} for i in range(0, len(calib), 32)])

        def get_next(self):
            return next(self.it, None)

    with tempfile.TemporaryDirectory() as tmp:
        pre = Path(tmp) / "pre.onnx"
        # symbolic shape inference trips on the option-A shortcut's Range; ONNX inference suffices
        quant_pre_process(str(fp32), str(pre), skip_symbolic_shape=True)
        quantize_static(str(pre), str(out), Reader(), quant_format=QuantFormat.QDQ,
                        per_channel=True, activation_type=QuantType.QUInt8,
                        weight_type=QuantType.QInt8)
    return out


def to_ort(path: Path) -> Path:
    """ORT format + the operator list a reduced ONNX Runtime build needs."""
    subprocess.run([sys.executable, "-m", "onnxruntime.tools.convert_onnx_models_to_ort",
                    str(path), "--optimization_style", "Fixed"], check=True, capture_output=True)
    path.with_suffix(".with_runtime_opt.ort").unlink(missing_ok=True)
    return path.with_suffix(".ort")


def evaluate(path: Path, x: np.ndarray, y: np.ndarray, output: int = 0, reps: int = 200) -> dict:
    """Accuracy on clean_test and single-image CPU latency under ONNX Runtime."""
    import onnxruntime as ort

    s = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    pred = s.run(None, {"images": x})[output]
    pred = pred.argmax(1) if pred.ndim == 2 else pred
    one = x[:1]
    s.run(None, {"images": one})
    t0 = time.perf_counter()
    for _ in range(reps):
        s.run(None, {"images": one})
    return {"accuracy": float((pred == y).mean()),
            "latency_ms": round((time.perf_counter() - t0) / reps * 1e3, 4)}


def tflite(nchw_onnx: Path, calib: np.ndarray, x: np.ndarray, y: np.ndarray, out: Path) -> dict:
    """onnx2tf in an isolated `uv` env, so TensorFlow never touches the project's pins."""
    script = Path(__file__).parent / "tflite_convert.py"
    with tempfile.TemporaryDirectory() as tmp:
        files = {}
        for name, arr in {"calib": calib.astype(np.float32), "x": x, "y": y}.items():
            files[name] = Path(tmp) / f"{name}.npy"
            np.save(files[name], np.ascontiguousarray(arr))
        cmd = ["uv", "run", "--isolated", "--no-project", "--python", "3.11",
               *(a for p in TFLITE_ENV for a in ("--with", p)), "python", str(script),
               str(nchw_onnx), str(files["calib"]), str(files["x"]), str(files["y"]), str(out)]
        r = subprocess.run(cmd, capture_output=True, text=True, check=False)
    results = out / "_tflite_results.json"
    if r.returncode or not results.exists():
        raise RuntimeError(f"TFLite conversion failed:\n{r.stderr[-2000:]}")
    data = json.loads(results.read_text())
    results.unlink()
    return data
