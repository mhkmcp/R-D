"""Artifact sidecar fields shared by every milestone (SPEC §11.3)."""

import hashlib
import json
import platform
import subprocess
import time

import numpy as np
import sklearn
import torch


def git_commit() -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True)
        dirty = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True,
                               check=False)
        return out.stdout.strip() + ("-dirty" if dirty.stdout.strip() else "")
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def config_hash(cfg: dict) -> str:
    return hashlib.sha256(json.dumps(cfg, sort_keys=True, default=str).encode()).hexdigest()[:16]


def sidecar(cfg: dict, seed: int, device: str) -> dict:
    return {
        "git_commit": git_commit(),
        "config_hash": config_hash(cfg),
        "config": cfg,
        "seed": seed,
        "device": device,
        "versions": {"python": platform.python_version(), "torch": torch.__version__,
                     "numpy": np.__version__, "sklearn": sklearn.__version__},
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
