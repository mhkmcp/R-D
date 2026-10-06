"""ONNX classifier → TFLite (fp32, fp16, full INT8) + C header. Runs in an isolated env.

Invoked by `export.py` through `uv run --isolated` so TensorFlow never enters the project env.
Usage: python tflite_convert.py <nchw.onnx> <calib.npy> <test_x.npy> <test_y.npy> <out_dir>
"""

import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np


def evaluate(path: Path, x: np.ndarray, y: np.ndarray) -> dict:
    from ai_edge_litert.interpreter import Interpreter

    it = Interpreter(model_path=str(path))
    it.allocate_tensors()
    inp, out = it.get_input_details()[0], it.get_output_details()[0]
    preds = []
    for k in range(len(x)):
        v = x[k:k + 1].astype(np.float32)
        if inp["dtype"] != np.float32:
            s, z = inp["quantization"]
            info = np.iinfo(inp["dtype"])
            v = np.clip(np.round(v / s + z), info.min, info.max).astype(inp["dtype"])
        it.set_tensor(inp["index"], v)
        it.invoke()
        preds.append(int(it.get_tensor(out["index"])[0].argmax()))
    s, z = inp["quantization"]
    return {"accuracy": float((np.array(preds) == y).mean()),
            "input": {"dtype": np.dtype(inp["dtype"]).name,
                      "shape": [int(d) for d in inp["shape"]],
                      "quantization": {"scale": float(s), "zero_point": int(z)}}}


def c_header(path: Path, name: str) -> str:
    data = path.read_bytes()
    rows = (", ".join(f"0x{b:02x}" for b in data[i:i + 12]) for i in range(0, len(data), 12))
    body = ",\n  ".join(rows)
    return (f"// {path.name}: {len(data)} bytes. Include once; keep in flash.\n"
            f"#include <stdint.h>\n"
            f"alignas(16) const uint8_t {name}[] = {{\n  {body}\n}};\n"
            f"const unsigned int {name}_len = {len(data)};\n")


def main(onnx_path, calib_path, x_path, y_path, out_dir) -> None:
    import onnx2tf
    from onnx2tf import onnx2tf as core

    calib = np.load(calib_path)
    # onnx2tf downloads a pickled sample for image inputs; use the clean calibration images instead
    core.download_test_image_data = lambda: calib[:20] / 255.0
    out_dir = Path(out_dir)
    with tempfile.TemporaryDirectory() as tmp:
        onnx2tf.convert(input_onnx_file_path=onnx_path, output_folder_path=tmp,
                        output_integer_quantized_tflite=True, quant_type="per-channel",
                        custom_input_op_name_np_data_path=[[
                            "images", calib_path, np.zeros((1, 1, 1, 3), np.float32),
                            np.ones((1, 1, 1, 3), np.float32)]],
                        non_verbose=True)
        stem = Path(onnx_path).stem
        picks = {"classifier_fp32.tflite": f"{stem}_float32.tflite",
                 "classifier_fp16.tflite": f"{stem}_float16.tflite",
                 "classifier_int8.tflite": f"{stem}_full_integer_quant.tflite"}
        for dst, src in picks.items():
            shutil.copy(Path(tmp) / src, out_dir / dst)
    x, y = np.load(x_path), np.load(y_path)
    results = {name: evaluate(out_dir / name, x, y) for name in picks}
    (out_dir / "classifier_int8_tflite.h").write_text(
        c_header(out_dir / "classifier_int8.tflite", "classifier_int8_tflite"))
    (out_dir / RESULTS).write_text(json.dumps(results))


RESULTS = "_tflite_results.json"

if __name__ == "__main__":
    main(*sys.argv[1:6])
    # TensorFlow and LiteRT in one process can abort in a mutex during interpreter teardown;
    # the results are on disk by now, so skip the teardown entirely
    sys.stdout.flush()
    os._exit(0)
