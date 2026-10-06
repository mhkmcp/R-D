# Image classifier with a built-in fault monitor

This folder holds a trained CIFAR-10 image classifier together with a monitor that watches the
model's internal activations. If the stored weights get corrupted, for example by bit flips in
memory, the monitor raises a flag. Everything needed is in `model.onnx`. You don't need the
training data, the training code or PyTorch.

## Install

Python 3.9 or newer, then:

```bash
pip install -r requirements.txt
python run.py --selftest        # should print "selftest PASSED"
```

The self-test checks that `model.onnx` is unaltered (via its SHA-256 in `manifest.json`). It then
checks that it reproduces the predictions and monitor decisions recorded at export.

## Use

```bash
python run.py cat.png ship.jpg                # class, confidence, monitor verdict
python run.py --alpha 0.001 photos/*.png      # stricter: fewer false alarms
python run.py --window 8 3 frames/*.png       # ALARM when 3 of the last 8 are flagged
python run.py --json photos/*.png             # machine-readable output
```

Images of any size are resized to 32×32 RGB. The model knows the 10 CIFAR-10 classes listed in
`manifest.json` (`classes`).

**What the output means**
- `monitor_score` is how unusual the model's internal behaviour looks. Higher means more
  unusual.
- `flagged` means the score is above the threshold calibrated for `--alpha`. On healthy hardware
  with normal images, about `alpha` of inferences are flagged anyway: 5 %, 1 % or 0.1 %.
- `ALARM` with `--window N M` means at least M of the last N inferences were flagged. Corrupted
  weights affect every inference, so a window catches them while ignoring isolated false alarms.

## From your own code

```python
import numpy as np, onnxruntime as ort
s = ort.InferenceSession("model.onnx")
logits, probs, pred, score, alarms = s.run(None, {"images": images_uint8_nhwc})
# alarms[:, k] corresponds to manifest["monitor"]["alphas"][k]
```

ONNX Runtime also runs on C/C++, C#, Java/Android, JavaScript (browser and Node) and iOS. The
same `model.onnx` works there, with input `images` as uint8 of shape (N, 32, 32, 3).

## Limits

- The monitor detects behaviour-changing weight corruption. It is not a defence against
  adversarial images.
- The false-alarm rate holds only for images like CIFAR-10. Unusual photos may be flagged for
  reasons unrelated to faults.
- The model comes from a single training run. `manifest.json` records its accuracy, the measured
  false-alarm rates and the latency the monitor adds on the export machine. Re-measure latency on
  your own device.
