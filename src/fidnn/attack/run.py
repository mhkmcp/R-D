"""`fidnn attack bfa`: run the L1 attacker and render the M-1b validation record (SPEC §12)."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from fidnn.attack import bfa
from fidnn.data.loader import Batches
from fidnn.data.prepare import load_arrays
from fidnn.models.checkpoint import load
from fidnn.provenance import sidecar
from fidnn.tables import md


def _tensor(x: np.ndarray, y: np.ndarray, mean, std) -> torch.Tensor:
    return torch.cat([xb for xb, _ in Batches(x, y, mean, std, 500)])


def run(model_id: str, seed: int, cfg_path: Path, data_cfg: Path, data_dir: Path,
        models_dir: Path, out_dir: Path, trials: int | None = None, log=print) -> dict:
    cfg = yaml.safe_load(cfg_path.read_text())
    trials = trials if trials is not None else cfg["trials"]
    arrays = load_arrays(data_cfg, data_dir)
    fit_x, fit_y = arrays.of("clean_fit")
    calib = [x for x, _ in Batches(fit_x[:cfg["calib_n"]], fit_y[:cfg["calib_n"]],
                                   arrays.mean, arrays.std, 64)]
    fp32 = load(model_id, seed, "fp32", models_dir)
    probes = _tensor(arrays.probe_x, arrays.probe_y, arrays.mean, arrays.std)
    probe_y = torch.from_numpy(arrays.probe_y)
    target = cfg["reference"]["target_accuracy"]

    curves = []
    for trial in range(trials):
        rng = np.random.default_rng(1000 * seed + trial)
        int8 = load(model_id, seed, "int8", models_dir, calib)   # fresh model per trial
        attack_idx = rng.choice(len(probe_y), size=cfg["attack_batch"], replace=False)
        eval_idx = rng.choice(len(probe_y), size=cfg["eval_n"], replace=False)
        clean_acc = bfa.accuracy(int8, probes[eval_idx], probe_y[eval_idx])
        log(f"trial {trial}: clean INT8 accuracy {clean_acc:.2f} %")
        df = bfa.attack(int8, fp32, probes[attack_idx], probe_y[attack_idx], cfg["max_flips"],
                        cfg["candidates_per_layer"], (probes[eval_idx], probe_y[eval_idx]),
                        log=log, stop_below=target)
        df.insert(0, "trial", trial)
        df.insert(1, "clean_accuracy", clean_acc)
        curves.append(df)
    curve = pd.concat(curves, ignore_index=True)

    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"bfa_{model_id}_seed{seed}"
    curve.to_parquet(out_dir / f"{stem}.parquet", index=False)
    n_flip = [int(g[g.accuracy < target].step.min()) if (g.accuracy < target).any() else None
              for _, g in curve.groupby("trial")]
    record = {"model": model_id, "precision": "int8", "attacker": "L1", "trials": trials,
              "n_flip": n_flip, "reference": cfg["reference"],
              **sidecar(cfg, seed=seed, device="cpu")}
    (out_dir / f"{stem}.json").write_text(json.dumps(record, indent=2, default=str))
    log(f"{stem}: N_flip {n_flip} vs published {cfg['reference']['n_flip_trials']}")
    return record


def write_report(art: Path, out: Path) -> None:
    """docs/M1b_bfa.md: our flip-budget-vs-accuracy curve against the published anchor."""
    sides = [json.loads(p.read_text()) for p in sorted(art.glob("bfa_*.json"))]
    if not sides:
        raise FileNotFoundError(f"no bfa_*.json under {art}")
    curves = pd.concat([pd.read_parquet(p) for p in sorted(art.glob("bfa_*.parquet"))],
                       ignore_index=True)
    side = sides[0]
    ref = side["reference"]
    ours = sorted(n for s in sides for n in s["n_flip"] if n is not None)
    never = sum(n is None for s in sides for n in s["n_flip"])
    published = sorted(ref["n_flip_trials"])
    lo, hi = published[0], published[-1]
    median = float(np.median(ours)) if ours else float("nan")
    verdict = (
        f"**M-1b gate PASSED.** Our N_flip over {len(ours)} trials is {ours} (median "
        f"{median:.0f}); the published range is {lo}–{hi}. The reimplementation collapses "
        f"ResNet-20 INT8 in a comparable flip budget."
        if ours and lo <= median <= hi else
        f"**M-1b gate NOT MET.** Our N_flip is {ours or 'never reached the target'} "
        f"(median {median:.0f}) against a published range of {lo}–{hi}"
        + (f"; {never} trial(s) never reached {ref['target_accuracy']} %." if never else ".")
        + " Explain the discrepancy or fix the implementation before generating faults (§12).")

    per_trial = (curves.groupby(["trial"])
                 .agg(clean_accuracy=("clean_accuracy", "first"), flips=("step", "max"),
                      final_accuracy=("accuracy", "last"), final_loss=("loss_after", "last"))
                 .reset_index())
    head = curves[curves.step <= 12].pivot_table(index="step", columns="trial", values="accuracy")
    layers = (curves.groupby("layer").size().sort_values(ascending=False)
              .rename("flips").reset_index().head(10))
    bits = curves.groupby("bit").size().rename("flips").reset_index()

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(f"""# M-1b — L1 (BFA) attacker validation

**Milestone:** M-1b (SPEC §12, §6.4) · **Model:** {side['model']} INT8 · **Commit:** `{side['git_commit']}`
· **Generated:** {side['timestamp']}

Generated by `fidnn attack bfa --report-only`; edit the code, not this file. The attacker sees only
a {side['config']['attack_batch']}-image sample, as in the paper; accuracy is measured on
{side['config']['eval_n']} held-out probe-pool images.

## Verdict

{verdict}

**Published anchor.** {ref['source']}
N_flip = {published} (5 trials), baseline {ref['baseline_accuracy']} %, quantised
{ref['quantised_accuracy']} %, target < {ref['target_accuracy']} %.

## Accuracy against flip budget (first 12 flips, per trial)

{md(head.reset_index(), floatfmt=".2f")}

## Per trial

{md(per_trial, floatfmt=".2f")}

## Where the attack flips bits

{md(layers)}

Bit positions (7 is the int8 sign bit):

{md(bits)}

## Reading the comparison

Differences from the paper that are expected, and are not by themselves implementation bugs:

- Our INT8 is post-training **per-channel** quantisation through qnnpack; the paper quantises
  weights **per layer** with its own scheme, so the int8 grid differs.
- The paper's baseline is its own trained ResNet-20; ours is trained on the 40k `train` split
  (SPEC §3.2), so the starting accuracy differs slightly.
- N_flip is a heavy-tailed quantity — the published trials themselves span {lo}–{hi}.
""")
    print(f"[m1b] wrote {out}")
