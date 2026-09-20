"""Load a trained checkpoint as an FP32 or INT8 model instance (SPEC §3, §4.4)."""

import json
from pathlib import Path

import torch
from torch import nn

from fidnn.models.quantize import quantize_ptq
from fidnn.models.registry import build


def checkpoint_path(model_id: str, seed: int, models_dir: Path) -> Path:
    return models_dir / f"{model_id}_seed{seed}.pt"


def load(model_id: str, seed: int, precision: str, models_dir: Path,
         calib: list[torch.Tensor] | None = None) -> nn.Module:
    """A fresh instance per call: the clean and fault instances must never be the same object."""
    model = build(model_id)
    model.load_state_dict(torch.load(checkpoint_path(model_id, seed, models_dir),
                                     map_location="cpu"))
    model = model.cpu().eval()
    if precision == "fp32":
        return model
    if not calib:
        raise ValueError("INT8 needs clean_fit calibration batches (SPEC §3.2)")
    return quantize_ptq(model, calib)


def manifest_entry(model_id: str, seed: int, models_dir: Path) -> dict:
    manifest = json.loads((models_dir / "manifest.json").read_text())
    return manifest["models"][f"{model_id}_seed{seed}"]
