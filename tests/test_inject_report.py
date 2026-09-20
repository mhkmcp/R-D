"""The M-2 exit record: gate verdict and per-arm Track S shares (SPEC §12)."""

import json

import numpy as np
import pandas as pd
import pytest

from fidnn.inject.report import GATE, write


def _arm(tmp_path, model, precision, s_share, n=400, seed=0):
    rng = np.random.default_rng(seed)
    labels = np.where(rng.random(n) < s_share,
                      rng.choice(["MASKED", "DEGRADED"], n),
                      rng.choice(["SDC", "CRASH"], n))
    df = pd.DataFrame({
        "injection_id": np.arange(n) // 8, "mode": "bf_w", "attacker": "L0",
        "bucket": rng.choice(["early", "middle", "late"], n),
        "stratum": rng.choice(["sign", "exp_msb"], n), "budget": rng.choice([1, 4, 16], n),
        "flips_applied": 1, "layers": "conv1", "probe_index": np.arange(n),
        "y": 0, "clean_pred": 0, "fault_pred": 1, "logit_shift": rng.random(n),
        "label": labels, "track": np.where(np.isin(labels, ["MASKED", "DEGRADED"]), "S", "H"),
    })
    stem = f"{model}_{precision}_bf_w_seed{seed}"
    df.to_parquet(tmp_path / f"{stem}_outcomes.parquet", index=False)
    (tmp_path / f"{stem}.json").write_text(json.dumps({
        "model": model, "precision": precision, "mode": "bf_w", "seed": seed,
        "injections": n // 8, "probes": n, "track_s_share": float((df.track == "S").mean()),
        "git_commit": "abc123", "timestamp": "2026-09-20T00:00:00+0600",
        "config": {"epsilon": 0.01, "probes_per_injection": 8},
    }))


def test_gate_passes_when_every_thesis_arm_clears_the_share(tmp_path):
    _arm(tmp_path, "m2", "fp32", 0.40)
    _arm(tmp_path, "m3", "int8", 0.30, seed=1)
    out = tmp_path / "M2_taxonomy.md"
    write(tmp_path, out)
    text = out.read_text()
    assert "M-2 gate PASSED" in text
    assert "m2" in text and "m3" in text and f"{GATE:.0%}" in text


def test_gate_fails_and_names_the_escalation_when_track_s_is_thin(tmp_path):
    _arm(tmp_path, "m2", "fp32", 0.40)
    _arm(tmp_path, "m3", "fp32", 0.02, seed=2)
    out = tmp_path / "M2_taxonomy.md"
    write(tmp_path, out)
    text = out.read_text()
    assert "M-2 gate NOT MET" in text
    assert "ResNet-56" in text


def test_m1_is_not_counted_towards_the_gate(tmp_path):
    _arm(tmp_path, "m1", "fp32", 0.0)        # smoke model: no thesis result, no gate weight
    _arm(tmp_path, "m2", "fp32", 0.50, seed=3)
    out = tmp_path / "M2_taxonomy.md"
    write(tmp_path, out)
    assert "M-2 gate PASSED" in out.read_text()


def test_report_needs_outcomes(tmp_path):
    with pytest.raises(FileNotFoundError):
        write(tmp_path, tmp_path / "M2_taxonomy.md")
