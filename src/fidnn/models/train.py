"""Classifier training on the `train` split and the M-1 accuracy gate (SPEC §3, §3.2, §12)."""

import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from torch import nn

from fidnn.data import cifar10
from fidnn.data.loader import Batches
from fidnn.data.prepare import load_prepared, subset_classes
from fidnn.models.quantize import quantize_ptq
from fidnn.models.registry import MODELS, build
from fidnn.provenance import sidecar


def _seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)




def make_scheduler(opt: torch.optim.Optimizer, cfg: dict, steps_per_epoch: int):
    total = cfg["epochs"] * steps_per_epoch
    if cfg["schedule"] == "multistep":
        return torch.optim.lr_scheduler.MultiStepLR(
            opt, [m * steps_per_epoch for m in cfg["milestones"]], gamma=0.1)
    return torch.optim.lr_scheduler.CosineAnnealingLR(opt, total)


@torch.no_grad()
def accuracy(model: nn.Module, batches: Batches) -> float:
    model.eval()
    correct = sum((model(x).argmax(1) == y).sum().item() for x, y in batches)
    return 100.0 * correct / len(batches.y)


def train(model_id: str, seed: int, device: str, model_cfg: Path, data_cfg: Path,
          data_dir: Path, out_dir: Path, epochs: int | None = None,
          log=print) -> dict:
    cfg = yaml.safe_load(model_cfg.read_text())
    if epochs is not None:
        cfg["epochs"] = epochs
    root = Path(yaml.safe_load(data_cfg.read_text())["root"])
    splits, meta = load_prepared(data_dir)
    mean, std = meta["normalisation"]["mean"], meta["normalisation"]["std"]

    kx, ky = cifar10.load_kaggle_train(root)
    tx, ty = cifar10.load_canonical(root, train=False)
    split = splits.split.to_numpy()
    fit_x, fit_y = kx[split == "clean_fit"], ky[split == "clean_fit"]
    kx, ky = kx[split == "train"], ky[split == "train"]
    classes = cfg.get("classes")
    kx, ky = subset_classes(kx, ky, classes)
    tx, ty = subset_classes(tx, ty, classes)
    fit_x, fit_y = subset_classes(fit_x, fit_y, classes)

    _seed_all(seed)
    model = build(model_id).to(device)
    train_b = Batches(kx, ky, mean, std, cfg["batch_size"], device, augment=True, shuffle=True,
                      seed=seed)
    opt = torch.optim.SGD(model.parameters(), lr=cfg["lr"], momentum=cfg["momentum"],
                          weight_decay=cfg["weight_decay"], nesterov=cfg.get("nesterov", False))
    sched = make_scheduler(opt, cfg, len(train_b))
    loss_fn = nn.CrossEntropyLoss()

    history = []
    t0 = time.time()
    for epoch in range(cfg["epochs"]):
        model.train()
        loss_sum, correct = 0.0, 0
        for x, y in train_b:
            opt.zero_grad(set_to_none=True)
            out = model(x)
            loss = loss_fn(out, y)
            loss.backward()
            opt.step()
            sched.step()
            loss_sum += loss.item() * len(y)
            correct += (out.argmax(1) == y).sum().item()
        history.append({"epoch": epoch + 1, "loss": loss_sum / len(ky),
                        "train_acc": 100.0 * correct / len(ky), "lr": sched.get_last_lr()[0]})
        log(f"{model_id} seed={seed} epoch {epoch + 1}/{cfg['epochs']} "
            f"loss {history[-1]['loss']:.4f} train_acc {history[-1]['train_acc']:.2f} "
            f"({time.time() - t0:.0f}s)")

    model = model.cpu().eval()
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{model_id}_seed{seed}"
    ckpt = out_dir / f"{stem}.pt"
    torch.save(model.state_dict(), ckpt)

    test_b = Batches(tx, ty, mean, std, 500, "cpu")
    calib = [x for x, _ in Batches(fit_x[:512], fit_y[:512], mean, std, 64, "cpu")]
    fp32_acc = accuracy(model, test_b)
    int8_acc = accuracy(quantize_ptq(model, calib), test_b)

    ref = cfg.get("reference") or {}
    result = {
        "model": model_id, "name": MODELS[model_id].name, "seed": seed,
        "test_acc_fp32": round(fp32_acc, 2), "test_acc_int8": round(int8_acc, 2),
        "reference": ref,
        "gate_pass": (abs(fp32_acc - ref["accuracy"]) <= 1.0) if ref.get("accuracy") else None,
        "train_n": len(ky), "test_n": len(ty),
        "checkpoint": str(ckpt), "sha256": cifar10.sha256(ckpt),
        "train_seconds": round(time.time() - t0), "train_device": device,
        "history": history,
        **sidecar(cfg, seed=seed, device="cpu"),
    }
    (out_dir / f"{stem}.json").write_text(json.dumps(result, indent=2))
    _update_manifest(out_dir / "manifest.json", stem, result, meta)
    log(f"{stem}: FP32 {fp32_acc:.2f} %  INT8 {int8_acc:.2f} %  gate {result['gate_pass']}")
    return result


def _update_manifest(path: Path, stem: str, result: dict, data_meta: dict) -> None:
    manifest = json.loads(path.read_text()) if path.exists() else {"models": {}}
    manifest["data"] = {"sha256": data_meta["sha256"], "integrity": data_meta["integrity"]}
    manifest["models"][stem] = {k: v for k, v in result.items() if k != "history"}
    path.write_text(json.dumps(manifest, indent=2))
