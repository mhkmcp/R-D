"""Render docs/M0_throughput.md from artifacts/m0/*.parquet."""

import json
from dataclasses import asdict, fields
from pathlib import Path

import pandas as pd

from fidnn.bench.budget import REINVESTMENTS, Assumptions, Measurements, Plan, ledger, plan_budget
from fidnn.bench.throughput import INT8_MPS_NOTE
from fidnn.models.registry import MODELS
from fidnn.tables import md

ARMS = [("cpu", "fp32"), ("cpu", "int8"), ("mps", "fp32"), ("mps", "int8")]


def _md(df: pd.DataFrame) -> str:
    return md(df, floatfmt="")


def _f(x: float, nd: int = 2) -> str:
    return "—" if pd.isna(x) else f"{x:,.{nd}f}"


def _throughput_table(inf: pd.DataFrame, model: str) -> str:
    rows = []
    q = inf[inf.model == model]
    for device, precision in ARMS:
        arm = q[(q.device == device) & (q.precision == precision)]
        if arm.empty:
            continue
        if (arm.status == "n/a").all():
            rows.append([f"{device.upper()} {precision.upper()}", "all", "n/a", "n/a", "n/a",
                         "n/a", "n/a"])
            continue
        base = {b: arm[(arm.hooks == "off") & (arm.batch == b)].median_ms.iloc[0] for b in (1, 32)}
        for r in arm[arm.batch == 32].itertuples():
            b1 = arm[(arm.batch == 1) & (arm.hooks == r.hooks) & (arm.tap_set == r.tap_set)]
            b1_ms = b1.median_ms.iloc[0]
            label = "off" if r.hooks == "off" else f"{r.hooks} / {r.tap_set} ({r.n_taps})"
            add32 = r.median_ms - base[32]
            rows.append([
                f"{device.upper()} {precision.upper()}", label,
                _f(b1_ms, 3), _f(r.median_ms), _f(r.samples_per_s, 0),
                "—" if r.hooks == "off" else f"+{_f(b1_ms - base[1], 3)} ms ({(b1_ms / base[1] - 1) * 100:+.0f} %)",
                "—" if r.hooks == "off" else f"+{_f(add32)} ms ({(r.median_ms / base[32] - 1) * 100:+.0f} %)",
            ])
    df = pd.DataFrame(rows, columns=["Arm", "Hooks / taps (K)", "bs=1 ms", "bs=32 ms",
                                     "bs=32 samples/s", "Added @ bs=1", "Added @ bs=32"])
    return _md(df)


def _feature_share(inf: pd.DataFrame) -> str:
    rows = []
    ok = inf[(inf.status == "ok") & (inf.batch == 32) & (inf.device == "cpu")]
    for (model, precision), g in ok.groupby(["model", "precision"]):
        off = g[g.hooks == "off"].median_ms.iloc[0]
        for tap_set in ("default", "extended"):
            cap = g[(g.hooks == "capture") & (g.tap_set == tap_set)].median_ms.iloc[0]
            feat = g[(g.hooks == "features") & (g.tap_set == tap_set)].median_ms.iloc[0]
            total = feat - off
            rows.append([model.upper(), precision.upper(), tap_set, _f(off), _f(cap - off),
                         _f(feat - cap), f"{(feat - cap) / total * 100:.0f} %" if total > 0 else "—"])
    return _md(pd.DataFrame(rows, columns=["Model", "Precision", "Taps", "Forward ms",
                                           "Hook (capture) ms", "Feature compute ms",
                                           "Feature share of monitor cost"]))


def _grad_table(grad: pd.DataFrame) -> str:
    rows = []
    for r in grad.itertuples():
        epoch_min = r.median_ms / 1e3 * Assumptions().train_images / r.batch / 60
        rows.append([r.model.upper(), r.device.upper(), r.kind, r.batch, _f(r.median_ms, 1),
                     _f(r.samples_per_s, 0), _f(epoch_min, 1) if r.kind == "train_step" else "—"])
    return _md(pd.DataFrame(rows, columns=["Model", "Device", "Step", "Batch", "ms / step",
                                           "samples/s", "min / epoch (50k)"]))


def _injection_table(inj: pd.DataFrame) -> str:
    rows = []
    for (model, precision), g in inj.groupby(["model", "precision"]):
        fr = g[g.kind == "flip_restore"]
        cks = g[g.kind == "checksum"].median_ms.iloc[0]
        rows.append([model.upper(), precision.upper(),
                     _f(fr[fr["count"] == 1].median_ms.mean(), 3),
                     _f(fr[fr["count"] == 16].median_ms.mean(), 3),
                     _f(fr.median_ms.max(), 3), _f(cks, 2)])
    return _md(pd.DataFrame(rows, columns=["Model", "Precision", "Flip+restore N=1 (ms, mean)",
                                           "N=16 (ms, mean)", "Worst layer (ms)",
                                           "state_dict checksum (ms)"]))


def _svdd_table(svdd: pd.DataFrame) -> str:
    s = svdd[svdd.data == "heavy"]
    exact = s[s.method == "ocsvm_exact"].groupby(["n", "d"]).agg(
        fit_s=("fit_s", "mean"), fit_max=("fit_s", "max"), n_sv=("n_sv", "max"),
        score=("score_per_s", "min")).reset_index()
    rows = [["exact OCSVM", r.n, r.d, _f(r.fit_s, 2), _f(r.fit_max, 2), r.n_sv, _f(r.score, 0)]
            for r in exact.itertuples()]
    ny = s[s.method == "nystroem_sgd"].groupby(["n", "d"]).agg(
        fit_s=("fit_s", "mean"), fit_max=("fit_s", "max"), score=("score_per_s", "min")).reset_index()
    rows += [["Nystroem(500) + SGD-OCSVM", r.n, r.d, _f(r.fit_s, 2), _f(r.fit_max, 2), "—",
              _f(r.score, 0)] for r in ny.itertuples()]
    return _md(pd.DataFrame(rows, columns=["Method", "n", "d", "Fit s (mean over ν)",
                                           "Fit s (worst ν)", "Max SVs", "Score samples/s"]))


def _plan_desc(p: Plan) -> str:
    return (f"{p.seeds} seeds · reps {p.reps} · taps {'extended' if p.extended_taps else 'default'}"
            f" · L2 {p.l2} · INT8 parity {'yes' if p.int8_parity else 'no'}"
            f" · M3 INT8 {'yes' if p.m3_int8 else 'CUT'}")


def _ledger_table(lg: pd.DataFrame) -> str:
    g = lg.groupby("line", sort=False).agg(hours=("hours", "sum"),
                                           slack=("hours_with_slack", "sum")).reset_index()
    g = g.sort_values("slack", ascending=False)
    g["share"] = g.slack / g.slack.sum() * 100
    rows = [[r.line, _f(r.hours, 1), _f(r.slack, 1), f"{r.share:.0f} %"] for r in g.itertuples()]
    rows.append(["**Total**", f"**{_f(g.hours.sum(), 1)}**", f"**{_f(g.slack.sum(), 1)}**", ""])
    return _md(pd.DataFrame(rows, columns=["Workload", "Measured h", "h × slack", "Share"]))


def _detail_table(lg: pd.DataFrame) -> str:
    rows = [[r.line, r.model.upper(), r.precision.upper(), _f(r.hours_with_slack, 2), r.detail]
            for r in lg.itertuples()]
    return _md(pd.DataFrame(rows, columns=["Workload", "Model", "Precision", "h × slack",
                                           "Basis"]))


def write(art: Path, out: Path, budget_hours: float = 250.0) -> None:
    inf = pd.read_parquet(art / "inference.parquet")
    grad = pd.read_parquet(art / "grad.parquet")
    inj = pd.read_parquet(art / "injection.parquet")
    svdd = pd.read_parquet(art / "svdd.parquet")
    side = json.loads((art / "sidecar.json").read_text())
    m = Measurements(inf, grad, inj, svdd)
    a = Assumptions()

    decision = plan_budget(m, budget_hours, a)
    base = ledger(Plan(), m, a)
    chosen = ledger(decision.plan, m, a)
    everything = Plan(seeds=10, extended_taps=True, l2="all", int8_parity=True)

    ok = inf[inf.status == "ok"]
    m2_cpu = ok[(ok.model == "m2") & (ok.device == "cpu") & (ok.batch == 32)]
    m2_feat_int8 = m2_cpu[(m2_cpu.precision == "int8") & (m2_cpu.hooks == "features")
                          & (m2_cpu.tap_set == "default")]
    m2_off_int8 = m2_cpu[(m2_cpu.precision == "int8") & (m2_cpu.hooks == "off")].median_ms.iloc[0]
    int8_feat_ratio = m2_feat_int8.median_ms.iloc[0] / m2_off_int8

    sweep = base[base.line == "Fault sweep (§4.2)"]
    sweep_min_per_arm_seed = sweep.hours.max() * 60 / Plan().seeds
    by_line = base.groupby("line").hours_with_slack.sum().sort_values(ascending=False)
    top_lines = " and ".join(f"{n} ({h:.0f} h)" for n, h in by_line.head(2).items())
    sweep_rank = list(by_line.index).index("Fault sweep (§4.2)") + 1
    exact = svdd[(svdd.method == "ocsvm_exact") & (svdd.data == "heavy")]
    worst_fit = exact[exact.n == exact.n.max()].fit_s.max()
    worst_fit_at = exact.loc[exact[exact.n == exact.n.max()].fit_s.idxmax()]

    sens = []
    for b in (100.0, 250.0, 500.0):
        d = plan_budget(m, b, a)
        sens.append([f"{b:.0f} h", d.mode, _plan_desc(d.plan), _f(d.total_hours, 1),
                     "; ".join(d.applied) or "—"])

    if decision.mode == "reinvest":
        verdict = (
            f"**§4.2 grid CONFIRMED as specified.** The unmodified SPEC workload costs "
            f"**{decision.baseline_hours:.0f} h** (with {a.slack}× slack) against a "
            f"{budget_hours:.0f} h budget, so no repetition, arm or model is cut, and every bit "
            f"stratum runs.\n\n**§3.3 reinvestment sized:** the plan funds "
            + ", ".join(f"**{x}**" for x in decision.applied)
            + f", for **{decision.total_hours:.0f} h** total."
        )
        if decision.declined:
            verdict += " Not funded: " + "; ".join(
                f"{n} (plan would total {h:.0f} h)" for n, h in decision.declined) + "."
    elif decision.mode == "cut":
        verdict = (f"**§4.2 grid REVISED.** Baseline {decision.baseline_hours:.0f} h exceeds the "
                   f"{budget_hours:.0f} h budget; cuts applied in §13 order: "
                   + "; ".join(decision.applied) + f". Plan total {decision.total_hours:.0f} h.")
    else:
        verdict = (f"**§4.2 grid DOES NOT FIT.** Even after every permitted cut "
                   f"({'; '.join(decision.applied)}) the plan needs {decision.total_hours:.0f} h "
                   f"against {budget_hours:.0f} h. Escalate: the SPEC does not permit further cuts.")

    mach = side["machine"]
    vers = side["versions"]
    quick_warn = (
        "\n> ⚠️ **Quick run (100 timed runs per cell).** Numbers are indicative; the M-0 exit "
        "record must come from `make m0` (≥1000 runs per cell, §9.4).\n" if side["quick"] else "")

    assumption_rows = [[f.name, getattr(a, f.name)] for f in fields(a)]
    reinvest_names = [v[0][0] for v in REINVESTMENTS]

    md = f"""# M-0 — Throughput calibration

**Milestone:** M-0 (SPEC §12) · **Generated:** {side['timestamp']} · **Commit:** `{side['git_commit']}`
· **Config hash:** `{side['config_hash']}`
{quick_warn}
M-0 is the gate SPEC §12 places before any sweep: measure samples/s for M1/M2/M3 on MPS and CPU,
FP32 and INT8, with hooks on and off; confirm or revise the §4.2 grid; and size the §3.3
reinvestment against the measurement. This page is generated by `fidnn bench m0` from
`artifacts/m0/*.parquet`; edit the code or `configs/bench/m0.yaml`, not this file.

## Verdict

{verdict}

| Budget | Outcome | Plan | Total h | Taken |
|---|---|---|---|---|
""" + "\n".join("| " + " | ".join(map(str, r)) + " |" for r in sens) + f"""

Everything in §3.3 at once (10 seeds, extended taps, L2 on every arm, INT8 parity) would cost
**{ledger(everything, m, a).hours_with_slack.sum():.0f} h**.

The decision rests on the measured unit costs below **and** on the workload assumptions listed at
the end — chiefly the L1 attack volume and the L2 cost factor, which SPEC does not pin. Changing
an assumption and re-running `fidnn bench m0 --report-only` re-derives the verdict.

## Machine (SPEC §9.4: specified in the thesis)

| | |
|---|---|
| Chip | {mach['chip']} |
| Memory | {mach['memory_gb']:.0f} GB |
| CPU cores | {mach['cpu_count']} (torch threads: {side['torch_threads']}) |
| OS | {mach['os']} |
| Python / torch / numpy / scikit-learn | {vers['python']} / {vers['torch']} / {vers['numpy']} / {vers['sklearn']} |
| Quantised engine | {side['quantized_engine']} |
| Inputs | {side['inputs']} |

## 1. Inference throughput — hooks off, capture-only, and full §5.2 features

Median latency per forward pass, absolute milliseconds first and percentage second, with the
baseline stated per row (§9.4: the denominator is the model). "capture" attaches the hooks and
keeps a reference to each tap output; "features" computes the 26-wide Blocks A–D descriptor per
tap (Block E is a cross-tap ratio added at fusion, and is negligible). INT8 × MPS is not
measurable: {INT8_MPS_NOTE}.
"""
    for model_id in ("m1", "m2", "m3"):
        spec = MODELS[model_id]
        role = "pipeline smoke test, no thesis result" if not spec.thesis_result else "primary"
        md += f"\n### {model_id.upper()} — {spec.name} ({role})\n\n{_throughput_table(inf, model_id)}\n"

    md += f"""
### Where the monitor's cost goes (CPU, batch 32)

{_feature_share(inf)}

## 2. Training and gradient steps (FP32, batch 128)

Sizes M-1 training (`train_step`) and the L1 BFA gradient ranking (`fwd_bwd`). Data loading and
augmentation are **not** included; the ledger's slack factor carries them.

{_grad_table(grad)}

## 3. Per-injection fixed costs (CPU)

At 32 probes per injection every injection is one batch, so fixed per-injection costs sit
directly beside the forward pass. INT8 weights are packed, so an INT8 flip is
unpack → XOR → requantise → repack, paid twice (flip and restore). The checksum covers the full
`state_dict()` including BN buffers (§4.4) and is amortised over {a.checksum_every} injections.

{_injection_table(inj)}

## 4. SVDD fit and scoring cost (§6.1)

Pessimistic synthetic features (Student-t, df = 3, per-feature scales over two decades); ν over
the §6.3 span {{0.001, 0.01, 0.1}}. d = 29 is one tap, 174 the 6-tap fused vector, 377 the 13-tap
fused vector.

{_svdd_table(svdd)}

## 5. Budget ledger — chosen plan

**Plan:** {_plan_desc(decision.plan)}

{_ledger_table(chosen)}

<details><summary>Per-arm detail</summary>

{_detail_table(chosen)}

</details>

### SPEC baseline (no reinvestment), for reference

{_ledger_table(base)}

## 6. Findings

1. **The §4.2 fault grid is not where the budget goes.** The full {_injections_str(a)}-injection
   grid costs at most **{sweep_min_per_arm_seed:.0f} CPU-minutes per arm per seed** (slowest arm,
   before slack), and ranks #{sweep_rank} of {len(by_line)} baseline lines. The largest are
   {top_lines} — both driven by volumes in §7 below that SPEC does not pin, so those assumptions
   deserve more scrutiny than the grid does.
2. **Feature extraction, not the forward pass, dominates monitored inference — especially in
   INT8.** M2 INT8 with the default 6 taps runs at {int8_feat_ratio:.1f}× the bare INT8 forward
   time (§1). The INT8 forward is fast, so the FP32 descriptor arithmetic (dequantise + quantiles +
   sorts) is the larger term. This matters for H3's < 10 % design target (§1), which will not be
   met by the unoptimised descriptor at K = 6; M-6's K sweep and a cheaper Block A (e.g. histogram
   quantiles) are the levers. It does not affect the budget verdict.
3. **INT8 × MPS is structurally absent**, as SPEC §3 anticipated. The overhead study's INT8 rows are
   CPU-only by necessity.
4. **Exact OCSVM cost at `clean_fit` scale.** The slowest measured fit at n = {int(exact.n.max()):,}
   is **{worst_fit:.1f} s** (d = {int(worst_fit_at.d)}, ν = {worst_fit_at.nu}) on the pessimistic
   synthetic features (§4). {"That is cheap enough that the §6.1 Nystroem path is not needed for cost, and its approximation-gap measurement becomes optional." if worst_fit < 120 else "That is expensive enough that the §6.1 Nystroem path should be planned, with its approximation gap measured on M2."}
   Real features may converge differently; re-check at M-4.
5. **qnnpack stores INT8-model biases in FP32.** `bf_b` on the INT8 arm therefore flips FP32 bias
   bits, not int8 ones. The INT8 `bf_b` results must say so; otherwise the INT8-vs-FP32 comparison
   for biases compares identical fault mechanics under two labels.
6. **The torch.ao FX quantisation flow is legacy** (torch ≥ 2.13 schedules its removal in favour
   of torchao). torch is pinned exactly ({vers['torch']}) in `pyproject.toml`; do not upgrade
   mid-thesis without re-running M-0.
7. **Memory figures are partial.** `tracemalloc` (named in §9.4) sees only Python-heap allocations,
   not torch's C++ allocator, so the CPU memory column understates tensor memory; MPS allocation
   is read from `torch.mps.current_allocated_memory()`. M-6 should add process RSS for CPU.
8. **Seen-vs-unseen split question, raised by M-0, resolved in SPEC §3.2.** The classifier trains
   on a 40,000-image `train` split; `clean_fit`/`clean_cal`/`clean_test` (6k/2k/2k) come from the
   other 10,000, so no detector record is an image the classifier saw. The costs are priced above:
   training at {a.train_images:,} images, SVDD fits at n = {a.svdd_n:,}.

## 7. Workload assumptions (not pinned by SPEC)

{_md(pd.DataFrame(assumption_rows, columns=["Assumption", "Value"]))}

Reinvestment priority (§3.3): {' → '.join(reinvest_names)}. Cut order (§4.2, §13): reinvestments →
repetitions 100 → 50 → M3's INT8 arm. Bit strata, `bf_w`/`bf_b` full grids and M2's INT8 arm are
not variables of the plan and cannot be cut (asserted in `fidnn.bench.budget`).

The ledger is serial wall-clock. CPU sweeps and MPS training could overlap in practice, so the
real calendar time is lower; the budget is not.
"""
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(md)
    (art / "budget.json").write_text(json.dumps({
        "budget_hours": budget_hours, "mode": decision.mode, "plan": asdict(decision.plan),
        "baseline_hours": decision.baseline_hours, "total_hours": decision.total_hours,
        "applied": decision.applied, "declined": decision.declined,
    }, indent=2))
    print(f"[m0] wrote {out}")


def _injections_str(a: Assumptions) -> str:
    from fidnn.bench.budget import _injections
    return f"{_injections(a, Plan().reps):,}"
