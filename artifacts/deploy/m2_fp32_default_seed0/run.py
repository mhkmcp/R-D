"""Run the exported classifier + fault monitor. Needs only: onnxruntime, numpy, Pillow.

    python run.py --ui                       # browser UI (needs nothing beyond Python itself)
    python run.py --selftest                 # verify this copy reproduces the export
    python run.py photo.png more/*.jpg       # classify images, flag suspicious inferences
    python run.py --alpha 0.001 --window 8 3 frames/*.png
    python run.py --model formats/classifier_int8.onnx photo.png   # classifier only, no monitor
"""

import argparse
import hashlib
import json
import sys
from collections import deque
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def serve_ui(host: str, port: int) -> int:
    """Static server rooted at the bundle (see `http.server`); the page does the inference."""
    import functools
    import http.server
    import webbrowser
    from typing import ClassVar

    class Handler(http.server.SimpleHTTPRequestHandler):
        extensions_map: ClassVar[dict] = {
            **http.server.SimpleHTTPRequestHandler.extensions_map,
            ".wasm": "application/wasm", ".mjs": "text/javascript",
            ".onnx": "application/octet-stream"}

        def log_message(self, *args):
            pass

    server = http.server.ThreadingHTTPServer(
        (host, port), functools.partial(Handler, directory=str(HERE)))
    url = f"http://{'localhost' if host in ('0.0.0.0', '') else host}:{server.server_port}/ui/"
    print(f"UI at {url}  (Ctrl+C to stop)")
    if host in ("0.0.0.0", ""):
        print("Other devices on this network: use this machine's IP address instead of localhost")
    webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


def load_images(paths: list[str]) -> np.ndarray:
    """Any Pillow-readable image → (N, 32, 32, 3) uint8 RGB; .npy files are taken as-is."""
    from PIL import Image

    out = []
    for p in paths:
        if p.endswith(".npy"):
            a = np.load(p)
            out.extend(a if a.ndim == 4 else a[None])
        else:
            img = Image.open(p).convert("RGB").resize((32, 32), Image.BILINEAR)
            out.append(np.asarray(img, dtype=np.uint8))
    return np.ascontiguousarray(np.stack(out).astype(np.uint8))


class Monitor:
    def __init__(self, model: Path | None = None, bundle: Path = HERE):
        import onnxruntime as ort

        self.manifest = json.loads((bundle / "manifest.json").read_text())
        model = model or bundle / self.manifest["files"]["model"]
        rel = model.resolve().relative_to(bundle).as_posix()
        expected = (self.manifest["sha256"]["model"] if rel == self.manifest["files"]["model"]
                    else self.manifest["formats"].get(rel, {}).get("sha256"))
        if expected and hashlib.sha256(model.read_bytes()).hexdigest() != expected:
            raise SystemExit(f"{model.name} does not match manifest.json — the file was altered")
        self.session = ort.InferenceSession(str(model), providers=["CPUExecutionProvider"])
        self.names = [o.name for o in self.session.get_outputs()]
        self.monitored = "score" in self.names
        self.alphas = self.manifest["monitor"]["alphas"]
        self.classes = self.manifest["classes"]

    def __call__(self, images: np.ndarray) -> dict:
        out = dict(zip(self.names, self.session.run(None, {"images": images})))
        if not self.monitored:
            z = out["logits"] - out["logits"].max(1, keepdims=True)
            out["probs"] = np.exp(z) / np.exp(z).sum(1, keepdims=True)
            out["pred"] = out["logits"].argmax(1)
        return out


def selftest(mon: Monitor) -> bool:
    ref = np.load(HERE / mon.manifest["files"]["check"])
    out = mon(ref["images"])
    tol = mon.manifest["check"]["score_tolerance"]
    ok = {"pred": bool((out["pred"] == ref["pred"]).all()),
          "alarms": bool((out["alarms"] == ref["alarms"]).all()),
          "score": bool(np.abs(out["score"] - ref["score"]).max() <= tol)}
    for k, v in ok.items():
        print(f"  {k:7s} {'OK' if v else 'MISMATCH'}")
    print("selftest", "PASSED" if all(ok.values()) else "FAILED")
    return all(ok.values())


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("images", nargs="*")
    ap.add_argument("--ui", action="store_true", help="serve the browser UI")
    ap.add_argument("--host", default="127.0.0.1", help="--ui: 0.0.0.0 to reach it from a phone")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--model", type=Path, help="another .onnx/.ort file from this bundle")
    ap.add_argument("--alpha", type=float, default=0.01,
                    help="clean false-alarm rate the threshold was calibrated for")
    ap.add_argument("--window", type=int, nargs=2, metavar=("N", "M"), default=(1, 1),
                    help="raise ALARM when M of the last N inferences are flagged")
    ap.add_argument("--json", action="store_true", help="one JSON object per image")
    args = ap.parse_args(argv)

    if args.ui:
        return serve_ui(args.host, args.port)
    mon = Monitor(args.model)
    if args.selftest:
        return 0 if selftest(mon) else 1
    if not args.images:
        ap.print_help()
        return 2
    if args.alpha not in mon.alphas:
        raise SystemExit(f"--alpha must be one of {mon.alphas}")
    k = mon.alphas.index(args.alpha)
    n, m = args.window
    history: deque[bool] = deque(maxlen=n)

    out = mon(load_images(args.images))
    for i, path in enumerate(args.images):
        row = {"image": path, "class": mon.classes[int(out["pred"][i])],
               "confidence": round(float(out["probs"][i].max()), 4)}
        if mon.monitored:
            flagged = bool(out["alarms"][i, k])
            history.append(flagged)
            row |= {"monitor_score": round(float(out["score"][i]), 4), "flagged": flagged,
                    "alarm": sum(history) >= m}
        if args.json:
            print(json.dumps(row))
            continue
        line = f"{path}: {row['class']} ({row['confidence']:.1%})"
        if mon.monitored:
            state = "ALARM" if row["alarm"] else ("flagged" if row["flagged"] else "ok")
            line += f"  monitor {row['monitor_score']:+.3f} → {state}"
        print(line)
    return 0


if __name__ == "__main__":
    sys.exit(main())
