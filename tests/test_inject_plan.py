"""SPEC §4.2: grid coverage, bit strata, and the replay record (§4.4)."""

import numpy as np
import pytest
import torch

from fidnn.inject.plan import REDUCED_STRATA, STRATA, cells, plan, rnd_val_plan
from fidnn.inject.targets import BUCKETS, for_mode, targets
from fidnn.models.quantize import quantize_ptq
from fidnn.models.registry import build


@pytest.fixture(scope="module")
def m2():
    return build("m2").eval()


@pytest.fixture(scope="module")
def m2_int8(m2):
    return quantize_ptq(m2, [torch.randn(8, 3, 32, 32)])


def test_strata_cover_every_bit_exactly_once():
    for width, groups in STRATA.items():
        bits = sorted(b for g in groups.values() for b in g)
        assert bits == list(range(width)), width
        assert all(s in groups for s in REDUCED_STRATA[width])


def test_full_grid_covers_every_cell_with_the_configured_repetitions(m2):
    p = plan(targets(m2, "m2"), "bf_w", seed=0, reps=5)
    covered = set(zip(p.bucket, p.stratum, p.budget))
    assert covered == set(cells("bf_w", 32, (1, 4, 16)))
    assert len(covered) == 3 * 5 * 3  # buckets × strata × budgets
    per_cell = p.groupby(["bucket", "stratum", "budget"]).injection_id.nunique()
    assert (per_cell == 5).all()


def test_budget_is_the_flip_count_and_flips_are_distinct(m2):
    p = plan(targets(m2, "m2"), "bf_w", seed=0, reps=3)
    per_injection = p.groupby("injection_id").size()
    assert set(per_injection) == {1, 4, 16}
    assert (p.groupby("injection_id").budget.first() == per_injection).all()
    pairs = p.groupby("injection_id").apply(
        lambda d: len({(lay, i, b) for lay, i, b in zip(d.layer, d.flat_index, d.bit)}),
        include_groups=False)
    assert (pairs == per_injection).all()


def test_bits_stay_inside_their_stratum_and_indices_inside_their_tensor(m2):
    p = plan(targets(m2, "m2"), "bf_w", seed=1, reps=2)
    sizes = {(t.layer, t.tensor): t.numel for t in targets(m2, "m2")}
    for row in p.itertuples():
        assert row.bit in STRATA[32][row.stratum]
        assert 0 <= row.flat_index < sizes[(row.layer, row.tensor)]


def test_reduced_modes_run_late_bucket_two_strata_and_two_budgets(m2):
    p = plan(targets(m2, "m2"), "bf_bn", seed=0, reps=2)
    assert set(p.bucket) == {"late"}
    assert set(p.stratum) == set(REDUCED_STRATA[32])
    assert set(p.budget) == {1, 16}
    assert set(p.tensor) == {"running_mean", "running_var"}


def test_int8_plan_uses_eight_bit_strata_on_weights(m2_int8):
    p = plan(targets(m2_int8, "m2"), "bf_w", seed=0, reps=2)
    assert set(p.storage) == {"qweight"}
    assert p.bit.max() <= 7
    assert set(p.stratum) == set(STRATA[8])


def test_int8_biases_are_fp32_under_qnnpack(m2_int8):
    p = plan(targets(m2_int8, "m2"), "bf_b", seed=0, reps=2)
    assert set(p.storage) == {"qbias"}
    assert set(p.stratum) == set(STRATA[32])  # 32-bit strata: qnnpack keeps biases in FP32


def test_every_bucket_has_targets(m2, m2_int8):
    for model, mode in ((m2, "bf_w"), (m2_int8, "bf_w"), (m2, "bf_bn")):
        buckets = {t.bucket for t in for_mode(targets(model, "m2"), mode)}
        assert buckets <= set(BUCKETS) and buckets


def test_rnd_val_corrupts_whole_parameters(m2):
    p = rnd_val_plan(targets(m2, "m2"), seed=0, reps=3)
    assert set(p.bucket) == {"late"} and set(p.budget) == {1, 16}
    per_injection = p.groupby("injection_id")
    assert (per_injection.flat_index.nunique() == per_injection.budget.first()).all()
    assert p.bit.between(0, 31).all()


def test_plan_is_deterministic_for_a_seed(m2):
    a = plan(targets(m2, "m2"), "bf_w", seed=7, reps=2)
    b = plan(targets(m2, "m2"), "bf_w", seed=7, reps=2)
    assert a.equals(b)
    assert not a.equals(plan(targets(m2, "m2"), "bf_w", seed=8, reps=2))


def test_vgg_buckets_fold_five_blocks_into_thirds():
    t = targets(build("m3").eval(), "m3")
    got = {name: {x.bucket for x in t if x.layer.startswith(name)}
           for name in ("block1", "block3", "block5", "classifier")}
    assert got == {"block1": {"early"}, "block3": {"middle"},
                   "block5": {"late"}, "classifier": {"late"}}


def test_records_carry_every_replay_field(m2):
    p = plan(targets(m2, "m2"), "bf_w", seed=0, reps=1)
    assert {"injection_id", "mode", "attacker", "bucket", "stratum", "budget", "layer", "tensor",
            "storage", "numel", "flat_index", "bit"} <= set(p.columns)
    assert p.flat_index.dtype == np.dtype("int64")
