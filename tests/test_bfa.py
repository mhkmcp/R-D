"""L1 BFA: surrogate fidelity, candidate ranking, progressive search, replay (SPEC §6.4, §12)."""

import json

import numpy as np
import pandas as pd
import pytest
import torch

from fidnn.attack import bfa
from fidnn.attack.run import write_report
from fidnn.inject.bitflip import state_checksum
from fidnn.models.quantize import quantize_ptq
from fidnn.models.registry import build


@pytest.fixture(scope="module")
def models():
    torch.manual_seed(0)
    fp32 = build("m2").eval()
    return fp32, quantize_ptq(fp32, [torch.randn(16, 3, 32, 32)])


@pytest.fixture(scope="module")
def probes():
    torch.manual_seed(1)
    return torch.randn(32, 3, 32, 32), torch.randint(0, 10, (32,))


def test_flipped_values_match_twos_complement_bytes():
    q = torch.tensor([0, 1, -1, 127, -128], dtype=torch.int8)
    assert bfa.flipped_values(q, 0).tolist() == [1, 0, -2, 126, -127]
    assert bfa.flipped_values(q, 7).tolist() == [-128, -127, 127, -1, 0]
    for bit in range(8):
        assert torch.equal(bfa.flipped_values(bfa.flipped_values(q, bit).to(torch.int8), bit),
                           q.to(torch.int16))


def test_surrogate_mirrors_the_int8_weights(models):
    fp32, int8 = models
    twin, convs = bfa.surrogate(fp32, int8)
    for name, conv in convs.items():
        mod = int8.get_submodule(name)
        assert torch.allclose(conv.weight.flatten(), mod.weight().dequantize().flatten())
        if mod.bias() is not None:
            assert torch.allclose(conv.bias, mod.bias())
    assert convs and all(c.weight.requires_grad for c in convs.values())
    import torch.nn.functional as F
    F.cross_entropy(twin(torch.randn(4, 3, 32, 32)), torch.randint(0, 10, (4,))).backward()
    ungraded = [n for n, c in convs.items() if c.weight.grad is None]
    assert not ungraded, f"cross-layer search would skip {ungraded}"


def test_surrogate_predictions_track_the_int8_model(models, probes):
    fp32, int8 = models
    twin, _ = bfa.surrogate(fp32, int8)
    x, _ = probes
    with torch.no_grad():
        agree = (twin(x).argmax(1) == int8(x).argmax(1)).float().mean()
    assert agree >= 0.75, f"surrogate agrees with INT8 on only {agree:.0%} of probes"


def test_element_scales_cover_every_weight(models):
    _, int8 = models
    mod = int8.get_submodule("layer1.0.conv1")
    scales = bfa.element_scales(mod)
    assert scales.numel() == mod.weight().numel()
    assert (scales > 0).all()


def test_candidates_are_loss_increasing_and_ranked(models):
    _, int8 = models
    mod = int8.get_submodule("layer3.2.conv2")
    grad = torch.randn(mod.weight().numel())
    cands = bfa.rank_candidates(mod, grad, k=5)
    assert 0 < len(cands) <= 5
    gains = [c[2] for c in cands]
    assert gains == sorted(gains, reverse=True) and all(g > 0 for g in gains)
    assert all(0 <= bit < 8 and 0 <= i < mod.weight().numel() for i, bit, _ in cands)


def test_attack_increases_loss_every_step(models, probes):
    fp32, int8 = quantize_fresh(models)
    x, y = probes
    df = bfa.attack(int8, fp32, x, y, budget=3, candidates_per_layer=3, log=lambda _: None)
    assert len(df) == 3
    assert (df.loss_after > df.loss_before).all()
    assert df.loss_before.is_monotonic_increasing


def quantize_fresh(models):
    """A clean INT8 twin: the calibration batch is fixed, so two calls give identical models."""
    fp32, _ = models
    torch.manual_seed(99)
    return fp32, quantize_ptq(fp32, [torch.randn(16, 3, 32, 32)])


def test_records_replay_the_attack_exactly(models, probes):
    fp32, int8 = quantize_fresh(models)
    x, y = probes
    df = bfa.attack(int8, fp32, x, y, budget=3, candidates_per_layer=3, log=lambda _: None)
    attacked = state_checksum(int8)

    _, replayed = quantize_fresh(models)
    assert state_checksum(replayed) != attacked
    bfa.replay(replayed, df)
    assert state_checksum(replayed) == attacked


def test_attack_stops_once_accuracy_falls_below_the_target(models, probes):
    fp32, int8 = quantize_fresh(models)
    x, y = probes
    df = bfa.attack(int8, fp32, x, y, budget=5, candidates_per_layer=2, eval_xy=(x, y),
                    log=lambda _: None, stop_below=101.0)  # trivially satisfied on step 1
    assert len(df) == 1 and "accuracy" in df


def _trial_rows(trial, n, final_acc):
    acc = np.linspace(90.0, final_acc, n)
    return pd.DataFrame({"trial": trial, "clean_accuracy": 90.0, "step": np.arange(1, n + 1),
                         "attacker": "L1", "layer": "layer3.2.conv2", "flat_index": 1,
                         "bit": 7, "loss_before": np.linspace(0.3, 5, n),
                         "loss_after": np.linspace(0.4, 5.1, n), "accuracy": acc})


def _write_run(tmp_path, n_flip, target=11.0):
    curves = pd.concat([_trial_rows(t, n, 9.0) for t, n in enumerate(n_flip)], ignore_index=True)
    curves.to_parquet(tmp_path / "bfa_m2_seed0.parquet", index=False)
    (tmp_path / "bfa_m2_seed0.json").write_text(json.dumps({
        "model": "m2", "precision": "int8", "attacker": "L1", "trials": len(n_flip),
        "n_flip": n_flip, "git_commit": "abc123", "timestamp": "2026-09-20T00:00:00+0600",
        "config": {"attack_batch": 128, "eval_n": 2000},
        "reference": {"target_accuracy": target, "n_flip_trials": [7, 10, 10, 12, 17],
                      "baseline_accuracy": 92.11, "quantised_accuracy": 92.28,
                      "source": "Rakin et al. 2019, Table 2"},
    }))


def test_report_passes_when_our_median_sits_in_the_published_range(tmp_path):
    _write_run(tmp_path, [8, 9, 11, 12, 14])
    out = tmp_path / "M1b_bfa.md"
    write_report(tmp_path, out)
    text = out.read_text()
    assert "M-1b gate PASSED" in text and "7–17" in text and "Rakin" in text


def test_report_flags_a_reimplementation_that_needs_far_more_flips(tmp_path):
    _write_run(tmp_path, [40, 38, 39, 41, 37])
    out = tmp_path / "M1b_bfa.md"
    write_report(tmp_path, out)
    text = out.read_text()
    assert "M-1b gate NOT MET" in text
    assert "before generating faults" in text
