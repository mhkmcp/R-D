A non-specialist now has two ways in: a browser app on your machine, and a shareable web page. Both use plain language ("Break the model's memory", "Wrong answer, silently") rather than thesis terms. The full suite passes (255 tests, 6 of them new), ruff is clean, and nothing is committed.

**1. Local app, live on the real model**
```bash
uv sync --group ui
uv run fidnn quantize m2   # once, so it starts fast
uv run fidnn ui            # opens http://127.0.0.1:7860
```
- **Step 1:** pick one of 24 CIFAR-10 pictures, or a random one.
- **Step 2:** choose where to damage the model (early, middle or late layers), which bit (from "sign bit, drastic" to "lowest bits, subtle"), and how many flips. Then press **Flip bits**, **Surprise me** or **Reset model**. Each flipped weight is shown with its real value before and after. An "Advanced" panel lets you flip one exact bit.
- **Step 3:** cards show the model's answer before and after, the detector verdict (🟢 / 🔴) at a false-alarm rate you choose, and what the fault did ("No effect", "Shaken, same answer", "Wrong answer, silently"). A per-layer chart shows where the damage shows up. "Try on 200 pictures" gives accuracy and alarm rate, with hidden and harmful faults kept separate.

I tested it on your M2 INT8 artifacts by driving every button through the live server. Each click takes 7–31 ms, and Reset brings the model back to its exact clean checksum. What the app shows is the same as running the full model; the tests check this. If an artifact is missing, the app lists the commands to run first instead of crashing.

**2. Shareable page:** [Bit-Flip Detector Lab](https://claude.ai/artifact/HuN2XkBdVpDy97xCpt6rnX)
- There's no live model behind it. It plays back 60 fault scenarios across the 24 pictures, precomputed from your real model with `fidnn ui --export`.
- Each flip shows the weight's 8 bits before and after, with the flipped bit marked.
- A second tab, "Results so far", shows how often each detector catches a fault, next to what monitoring costs in time. Detection is never shown without its cost (C5).
- I only checked that the page's script parses; I haven't viewed it rendered. Please look at it on a phone width and in dark mode.
- It's private. Supervisors or examiners can't open it until you share it from the page's Share menu.

**What viewers will see**
- With your current single-seed results, the inner-layer detector almost never fires. Across the 60 scenarios × 24 pictures it flagged 1 pair at the 1% false-alarm rate. On harmful faults, the answer-only baselines catch far more (MSP about 30% vs D2 about 3%).
- The page states this directly, together with a banner saying the numbers come from 1 seed and the thesis needs at least 5. If you'd rather lead with a different framing for a supervisor meeting, tell me.

**One bug in the existing results file, unrelated to the demo:** the M-5 headline parquet has a `precision` column holding `int8`. The classification-precision metric appears to have been overwritten by the model-precision column when `fidnn eval` built that table. The demo doesn't use that column, but `fidnn eval` should be fixed before those tables go in the thesis.

New code is in [src/fidnn/demo/](src/fidnn/demo/README.md): the session logic, the Gradio app and the exporter, with tests in [tests/test_demo.py](tests/test_demo.py). I also pulled the planner's sampling into a `sample_flips` helper; the existing planner's output is byte-identical to before. Gradio is an optional install (`uv sync --group ui`); the torch pin is unchanged. The `Readme.md` has a short "Try the demo" section.