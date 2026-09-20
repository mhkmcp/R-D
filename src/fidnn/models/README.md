# fidnn.models

Model builders and INT8 quantisation. SPEC §3.

| ID | Builder | Classes | Thesis result |
|---|---|---|---|
| `m1` | `resnet8` | 2 | No — pipeline smoke test |
| `m2` | `resnet20` | 10 | Yes |
| `m3` | `vgg11_bn` | 10 | Yes |

`registry.build(model_id)` is the only entry point callers should use.

## Design

- **`Tap()`** (`common.py`) returns a plain `nn.Identity`. It costs nothing with no hook attached, and
  it survives FX INT8 conversion under its original name, so one tap registry addresses FP32 and INT8
  models alike. It must **not** be a subclass: the FX tracer only treats `torch.nn` modules as leaves
  and would trace through a subclass and erase it.
- **ResNet** (`resnet_cifar.py`): He et al. 2016 CIFAR variant, 3 stages × n blocks, 16/32/64
  channels, option-A shortcuts (zero-padded identity, stride-2 subsampling), so ResNet-20 ≈ 0.27M
  params. Each block has `tap_pre` (residual branch, before the add) and `tap_out` (after the
  post-add ReLU); the H4 analysis needs both. The add goes through `FloatFunctional` so it quantises.
- **VGG-11-BN** (`vgg.py`): 8 conv+BN+ReLU layers in 5 max-pool blocks, then the 3-layer FC head
  (512-512-10, dropout) of the pinned reference, so the M-1 gate compares like with like. No skip
  connections. That's the plain half of the M2-vs-M3 contrast. Taps: `blockB.tapK` after each conv
  (extended set), `blockB.tap_out` after each pool (default set). `CFG` is imported by the tap registry.
- **INT8** (`quantize.py`): torch.ao FX post-training static quantisation, qnnpack backend (the only
  quantised engine on Apple silicon). That flow is legacy from torch 2.13, which is why torch is pinned
  exactly in `pyproject.toml`. INT8 kernels are **CPU-only**, with no MPS path.
  `quantize_ptq` returns a copy and never mutates its input.

## Training (`train.py`, M-1)

`fidnn train m2 --device mps --seed 0` trains on the 40k `train` split (SPEC §3.2), then evaluates on **CPU**: FP32 accuracy and INT8 PTQ
accuracy (calibrated on 512 `clean_fit` images) on the canonical probe pool. It writes
`artifacts/models/{model}_seed{seed}.{pt,json}` and upserts `manifest.json`.

| Model | Config | Recipe | Reference |
|---|---|---|---|
| m1 | `configs/model/resnet8.yaml` | cat vs dog, 30 epochs cosine | none (smoke test) |
| m2 | `configs/model/resnet20.yaml` | He et al.: lr 0.1, ÷10 at epochs 82/123, 164 epochs, wd 1e-4 | 91.25 % |
| m3 | `configs/model/vgg11bn.yaml` | lr 0.05 Nesterov, cosine, 200 epochs, wd 5e-4 | 92.79 % (chenyaofo) |

- Training runs on MPS for speed. Every *reported* number (accuracy, and everything downstream) is
  computed on CPU, and the checkpoint is saved from CPU.
- The test set is evaluated once, at the end, and never used for early stopping or tuning.
- `gate_pass` = FP32 accuracy within 1 pt of the reference. Both references trained on all 50k
  images; ours sees 40k, which costs an expected ≈ 0.3–0.8 pt (`docs/Dataset.md` §5).

## Invariants

- Calibrate INT8 on a `clean_fit` subsample only (SPEC §3.2). Random inputs are acceptable only for
  timing (M-0).
- Adding a tap: add a `Tap()` module in the model, then register it in `fidnn.taps.registry`. Every
  model needs ≥ 5 usable taps.
- Adding a model: builder + `MODELS` entry + tap registry entry + reference accuracy in
  `artifacts/models/manifest.json`. Reinstating M4/M5 is a C0 addition and needs sign-off.
- M-1 gate: trained accuracy within 1 pt of reference (ResNet-20 91.25 %; VGG-11-BN pinned source).
