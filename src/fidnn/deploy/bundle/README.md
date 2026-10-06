# Image classifier with a built-in fault monitor: portable bundle

This folder holds a trained CIFAR-10 image classifier and a monitor that watches the model's
internal activations. If the stored weights get corrupted, for example by bit flips in memory,
the monitor raises a flag. You don't need the training data, the training code or PyTorch.

## Pick the file for your device

| Device | File | Monitor? |
|---|---|---|
| Any device with a browser (laptop, phone, tablet) | `ui/` running `model.onnx` | yes |
| Linux, Windows, macOS, Raspberry Pi, Jetson (Python or C/C++) | `model.onnx` | yes |
| Android, iOS, ONNX Runtime minimal builds | `formats/model.ort` | yes |
| Same, classifier only, about 2× faster and 3× smaller | `formats/classifier_int8.onnx` / `.ort` | no |
| Android / Linux with LiteRT (TFLite) | `formats/classifier_fp32.tflite`, `classifier_fp16.tflite` | no |
| Microcontrollers (ESP32-S3, STM32H7, Cortex-M7) via TFLite Micro | `formats/classifier_int8.tflite`, or `classifier_int8_tflite.h` compiled into the firmware | no |

`manifest.json` → `formats` lists each file with its accuracy on 2,000 held-out test pictures,
its size, its SHA-256 and its runtime. The model with the monitor needs more compute. The
manifest records how much time the monitor adds on the export machine (`overhead`). Re-measure it
on your own device.

## Browser UI

```bash
python run.py --ui                    # opens http://localhost:8000/ui/
python run.py --ui --host 0.0.0.0     # reach it from a phone on the same network
```

You only need Python's standard library to serve it. The inference runs inside the browser with
ONNX Runtime Web, which is included in `ui/`, so it works offline. Any static web server rooted
at this folder works too. Opening `ui/index.html` by double-clicking it does not: browsers block
loading the model from `file://`.

In the UI you can:
- pick a test picture or upload or photograph your own, and see the answer and the monitor's
  verdict;
- choose how many false alarms are acceptable: 5 %, 1 % or 0.1 % of pictures on a healthy model;
- **corrupt the model's memory**: flip a chosen bit in random weights of a chosen layer. You then
  see whether the answer changes and whether the monitor notices. The file on disk is never
  touched.

## Command line

Install with `pip install -r requirements.txt` (onnxruntime, numpy, Pillow), then:

```bash
python run.py --selftest                          # checks model.onnx is unaltered and matches
python run.py cat.png ship.jpg                    # class, confidence, monitor verdict
python run.py --alpha 0.001 photos/*.png          # stricter: fewer false alarms
python run.py --window 8 3 frames/*.png           # ALARM when 3 of the last 8 are flagged
python run.py --model formats/classifier_int8.onnx cat.png   # any .onnx/.ort file here
```

## From your own code

**ONNX Runtime (Python, C, C++, C#, Java, JavaScript).** Input `images` is uint8 of shape
(N, 32, 32, 3), RGB, 0–255. Normalisation is built into the model.

```python
import onnxruntime as ort
s = ort.InferenceSession("model.onnx")
logits, probs, pred, score, alarms = s.run(None, {"images": images_uint8_nhwc})
# alarms[:, k] corresponds to manifest["monitor"]["alphas"][k]
```

**TFLite / LiteRT.** Input is (1, 32, 32, 3) RGB, pixel values 0–255:
- float32 files take raw 0–255 floats;
- the INT8 file takes int8, quantised with the input tensor's scale and zero point:
  `q = round(pixel / scale + zero_point)`. `manifest.json` records both.

**Microcontrollers.**
1. Add `classifier_int8_tflite.h` to the firmware.
2. Create a `tflite::MicroInterpreter` over `classifier_int8_tflite`.
3. Register Conv2D, Add, Pad, StridedSlice, Mean, FullyConnected, Reshape, Quantize and
   Dequantize.
4. Give it a tensor arena of about 100 KB, and increase it until `AllocateTensors()` succeeds.

## Limits

- **Microcontrollers get the classifier only.** The monitor is too heavy for them; it needs about
  1 MB and sorts every activation of the first layer.
- **Not an adversarial defence.** The monitor detects weight corruption that changes the model's
  behaviour. It is not a defence against adversarial images.
- **CIFAR-10-like images only.** The false-alarm rate holds only for images like CIFAR-10.
  Unusual photos may be flagged for reasons unrelated to faults.
- **One training run.** The model comes from a single training seed. It is a demonstration, not a
  research result.
