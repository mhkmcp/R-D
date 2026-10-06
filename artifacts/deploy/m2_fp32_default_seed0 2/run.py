"""Run the exported classifier + fault monitor. Needs only: onnxruntime, numpy, Pillow.

    python run.py --selftest                 # verify this copy reproduces the export
    python run.py photo.png more/*.jpg       # classify images, flag suspicious inferences
    python run.py --alpha 0.001 --window 8 3 frames/*.png
"""

import argparse
import hashlib
import json
import sys
from collections import deque
from pathlib import Path

import numpy as np
import onnxruntime as ort

HERE = Path(__file__).resolve().parent


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
    def __init__(self, bundle: Path = HERE):
        self.manifest = json.loads((bundle / "manifest.json").read_text())
        model = bundle / self.manifest["files"]["model"]
        digest = hashlib.sha256(model.read_bytes()).hexdigest()
        if digest != self.manifest["sha256"]["model"]:
            raise SystemExit(f"{model.name} does not match manifest.json — the file was altered")
        self.session = ort.InferenceSession(str(model), providers=["CPUExecutionProvider"])
        self.alphas = self.manifest["monitor"]["alphas"]
        self.classes = self.manifest["classes"]

    def __call__(self, images: np.ndarray) -> dict:
        logits, probs, pred, score, alarms = self.session.run(None, {"images": images})
        return {"logits": logits, "probs": probs, "pred": pred, "score": score, "alarms": alarms}


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
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--alpha", type=float, default=0.01,
                    help="clean false-alarm rate the threshold was calibrated for")
    ap.add_argument("--window", type=int, nargs=2, metavar=("N", "M"), default=(1, 1),
                    help="raise ALARM when M of the last N inferences are flagged")
    ap.add_argument("--json", action="store_true", help="one JSON object per image")
    args = ap.parse_args(argv)

    mon = Monitor()
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
        flagged = bool(out["alarms"][i, k])
        history.append(flagged)
        alarm = sum(history) >= m
        row = {"image": path, "class": mon.classes[int(out["pred"][i])],
               "confidence": round(float(out["probs"][i].max()), 4),
               "monitor_score": round(float(out["score"][i]), 4), "flagged": flagged,
               "alarm": alarm}
        if args.json:
            print(json.dumps(row))
        else:
            state = "ALARM" if alarm else ("flagged" if flagged else "ok")
            print(f"{path}: {row['class']} ({row['confidence']:.1%})  "
                  f"monitor {row['monitor_score']:+.3f} → {state}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
