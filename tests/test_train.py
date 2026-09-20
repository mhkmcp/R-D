import numpy as np
import torch

from fidnn.data.loader import Batches
from fidnn.data.prepare import subset_classes
from fidnn.models.registry import build
from fidnn.models.train import accuracy, make_scheduler


def test_model_classes_reads_the_subset_from_the_config():
    """M1 is cat-vs-dog; everything downstream must restrict its probes the same way."""
    from fidnn.data.prepare import model_classes
    from fidnn.models.registry import config_path
    assert model_classes(config_path("m1")) == [3, 5]
    assert model_classes(config_path("m2")) is None


def test_subset_remaps_labels():
    x = np.arange(10)[:, None]
    y = np.arange(10)
    xs, ys = subset_classes(x, y, [3, 5])
    assert xs.ravel().tolist() == [3, 5] and ys.tolist() == [0, 1]
    assert subset_classes(x, y, None)[1] is y


def test_multistep_schedule_matches_he_et_al():
    opt = torch.optim.SGD([torch.zeros(1, requires_grad=True)], lr=0.1)
    cfg = {"epochs": 164, "schedule": "multistep", "milestones": [82, 123]}
    sched = make_scheduler(opt, cfg, steps_per_epoch=391)
    lrs = []
    for _ in range(164 * 391):
        opt.step()
        sched.step()
        lrs.append(sched.get_last_lr()[0])
    assert np.isclose(lrs[82 * 391 - 2], 0.1) and np.isclose(lrs[82 * 391], 0.01)
    assert np.isclose(lrs[-1], 0.001)


def test_a_few_steps_reduce_loss_and_accuracy_is_bounded():
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    x = rng.integers(0, 256, (64, 32, 32, 3), dtype=np.uint8)
    y = rng.integers(0, 2, 64)
    model = build("m1")
    b = Batches(x, y, [0.5] * 3, [0.25] * 3, batch_size=64, augment=True, shuffle=True)
    opt = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9)
    losses = []
    for _ in range(15):
        for xb, yb in b:
            model.train()
            loss = torch.nn.functional.cross_entropy(model(xb), yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())
    assert losses[-1] < losses[0]
    assert 0.0 <= accuracy(model, Batches(x, y, [0.5] * 3, [0.25] * 3, 32)) <= 100.0
