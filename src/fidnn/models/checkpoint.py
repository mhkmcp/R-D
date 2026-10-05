"""Load a trained checkpoint as an FP32 or INT8 model instance (SPEC §3, §4.4)."""

import hashlib
import json
from pathlib import Path

import torch
import yaml
from torch import nn

from fidnn.inject.bitflip import state_checksum
from fidnn.models.quantize import quantize_ptq
from fidnn.models.registry import build


class StaleInt8Error(RuntimeError):
    """The persisted INT8 model does not match its sidecar or the calibration set asked for."""


def checkpoint_path(model_id: str, seed: int, models_dir: Path) -> Path:
    return models_dir / f"{model_id}_seed{seed}.pt"


def int8_path(model_id: str, seed: int, models_dir: Path) -> Path:
    return models_dir / f"{model_id}_seed{seed}_int8.pt"


def calib_hash(calib: list[torch.Tensor]) -> str:
    return hashlib.sha256(torch.cat(calib).contiguous().numpy().tobytes()).hexdigest()[:16]


def _fp32(model_id: str, seed: int, models_dir: Path) -> nn.Module:
    model = build(model_id)
    model.load_state_dict(torch.load(checkpoint_path(model_id, seed, models_dir),
                                     map_location="cpu"))
    return model.cpu().eval()


def _load_int8(fp32: nn.Module, path: Path, calib: list[torch.Tensor] | None) -> nn.Module:
    """Skeleton from an uncalibrated PTQ pass, then the stored qparams and packed weights."""
    record = json.loads(path.with_suffix(".json").read_text())
    if calib and calib_hash(calib) != record["calib_hash"]:
        raise StaleInt8Error(f"{path.name} was calibrated on a different clean_fit subset")
    model = quantize_ptq(fp32, [torch.zeros(1, 3, 32, 32)])
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=False))
    if state_checksum(model) != record["checksum"]:
        raise StaleInt8Error(f"{path.name} does not reproduce its recorded checksum")
    return model


def load(model_id: str, seed: int, precision: str, models_dir: Path,
         calib: list[torch.Tensor] | None = None) -> nn.Module:
    """A fresh instance per call: the clean and fault instances must never be the same object.

    INT8 comes from `{model}_seed{n}_int8.pt` when `fidnn quantize` has written it, else from PTQ.
    """
    model = _fp32(model_id, seed, models_dir)
    if precision == "fp32":
        return model
    stored = int8_path(model_id, seed, models_dir)
    if stored.exists():
        return _load_int8(model, stored, calib)
    if not calib:
        raise ValueError("INT8 needs clean_fit calibration batches (SPEC §3.2)")
    return quantize_ptq(model, calib)


def save_int8(model_id: str, seed: int, models_dir: Path, calib: list[torch.Tensor],
              extra: dict | None = None) -> dict:
    """Run PTQ once and persist the INT8 `state_dict` with a checksum sidecar (SPEC §11.3)."""
    model = quantize_ptq(_fp32(model_id, seed, models_dir), calib)
    path = int8_path(model_id, seed, models_dir)
    torch.save(model.state_dict(), path)
    record = {"model": model_id, "seed": seed, "precision": "int8",
              "checksum": state_checksum(model), "calib_hash": calib_hash(calib),
              "calib_n": int(sum(len(x) for x in calib)), **(extra or {})}
    path.with_suffix(".json").write_text(json.dumps(record, indent=2, default=str))
    reloaded = _load_int8(_fp32(model_id, seed, models_dir), path, calib)
    if state_checksum(reloaded) != record["checksum"]:
        raise StaleInt8Error(f"{path.name} does not round-trip")
    return record


def quantize(model_id: str, seed: int, cfg_path: Path, data_cfg: Path, data_dir: Path,
             models_dir: Path, log=print) -> dict:
    """`fidnn quantize`: the persisted INT8 model every later command loads."""
    from fidnn.data.loader import Batches
    from fidnn.data.prepare import load_arrays, model_classes, subset_classes
    from fidnn.models.registry import config_path
    from fidnn.provenance import sidecar

    cfg = yaml.safe_load(cfg_path.read_text())
    arrays = load_arrays(data_cfg, data_dir)
    fit_x, fit_y = subset_classes(*arrays.of("clean_fit"), model_classes(config_path(model_id)))
    calib = [x for x, _ in Batches(fit_x[:cfg["calib_n"]], fit_y[:cfg["calib_n"]],
                                   arrays.mean, arrays.std, 64)]
    record = save_int8(model_id, seed, models_dir, calib,
                       extra=sidecar({"calib_n": cfg["calib_n"]}, seed=seed, device="cpu"))
    log(f"{int8_path(model_id, seed, models_dir)}: checksum {record['checksum']}")
    return record


def manifest_entry(model_id: str, seed: int, models_dir: Path) -> dict:
    manifest = json.loads((models_dir / "manifest.json").read_text())
    return manifest["models"][f"{model_id}_seed{seed}"]
