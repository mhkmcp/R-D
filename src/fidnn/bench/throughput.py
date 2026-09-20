"""M-0 measurement harness (SPEC §12, protocol §9.4). See bench/README.md."""

import hashlib
import json
import platform
import random
import subprocess
import time
import tracemalloc
import warnings
from collections.abc import Callable
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
import torch
import yaml
from torch import nn

from fidnn.inject.bitflip import flip_bits_, flip_quantized_weight_, state_checksum
from fidnn.models.quantize import BACKEND, quantize_ptq
from fidnn.models.registry import MODELS, build
from fidnn.taps.hooks import TapMonitor
from fidnn.taps.registry import taps

INT8_MPS_NOTE = "PyTorch quantised kernels are CPU-only (qnnpack); no MPS INT8 path (SPEC §3)"


def _sync(device: str) -> None:
    if device == "mps":
        torch.mps.synchronize()


def time_fn(fn: Callable[[], object], device: str, warmup: int, runs: int) -> np.ndarray:
    """Per-call wall time in seconds; synchronises the device before every timestamp."""
    for _ in range(warmup):
        fn()
    _sync(device)
    out = np.empty(runs)
    for i in range(runs):
        _sync(device)
        t0 = time.perf_counter()
        fn()
        _sync(device)
        out[i] = time.perf_counter() - t0
    return out


def _summary(t: np.ndarray, per_call_samples: int) -> dict:
    med = float(np.median(t))
    return {
        "runs": len(t),
        "median_ms": med * 1e3,
        "p10_ms": float(np.percentile(t, 10)) * 1e3,
        "p90_ms": float(np.percentile(t, 90)) * 1e3,
        "samples_per_s": per_call_samples / med,
    }


def _seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# --------------------------------------------------------------------------- timed bodies


def _forward(model: nn.Module, x: torch.Tensor) -> None:
    with torch.no_grad():
        model(x)


def _fwd_bwd(model: nn.Module, loss_fn: nn.Module, x: torch.Tensor, y: torch.Tensor) -> None:
    model.zero_grad(set_to_none=True)
    loss_fn(model(x), y).backward()


def _train_step(fwd_bwd: Callable[[], None], opt: torch.optim.Optimizer) -> None:
    fwd_bwd()
    opt.step()


def _fp32_flip(rng: np.random.Generator, weights: list[torch.Tensor], count: int) -> None:
    w = weights[rng.integers(len(weights))]
    idx = rng.integers(w.numel(), size=count).tolist()
    bits = rng.integers(32, size=count).tolist()
    flip_bits_(w.data, idx, bits)
    flip_bits_(w.data, idx[::-1], bits[::-1])


def _int8_flip(rng: np.random.Generator, mod: nn.Module, numel: int, count: int) -> None:
    idx = rng.integers(numel, size=count).tolist()
    bits = rng.integers(8, size=count).tolist()
    flip_quantized_weight_(mod, idx, bits)
    flip_quantized_weight_(mod, idx[::-1], bits[::-1])


# --------------------------------------------------------------------------- inference matrix


def _variants(model_id: str, cfg: dict, calib: list[torch.Tensor]):
    """Yield (device, precision, model); model is None where the arm does not exist (INT8 × MPS)."""
    fp32 = build(model_id).eval()
    int8 = quantize_ptq(fp32, calib)
    for device in cfg["devices"]:
        yield device, "fp32", fp32.to(device)
        yield device, "int8", (int8 if device == "cpu" else None)


def measure_inference(cfg: dict, timing: dict, log: Callable[[str], None]) -> pd.DataFrame:
    rows = []
    for model_id in cfg["models"]:
        calib = [torch.randn(32, 3, 32, 32) for _ in range(4)]
        for device, precision, model in _variants(model_id, cfg, calib):
            for bs in cfg["batch_sizes"]:
                x = torch.randn(bs, 3, 32, 32, device=device)
                for hooks in cfg["hook_modes"]:
                    for tap_set in (["-"] if hooks == "off" else cfg["tap_sets"]):
                        row = {"model": model_id, "device": device, "precision": precision,
                               "batch": bs, "hooks": hooks, "tap_set": tap_set}
                        tap_list = taps(model_id, tap_set) if hooks != "off" else []
                        row["n_taps"] = len(tap_list)
                        if model is None:
                            rows.append({**row, "status": "n/a", "note": INT8_MPS_NOTE})
                            continue
                        mon = TapMonitor(model, tap_list, hooks) if hooks != "off" else None

                        fwd = partial(_forward, model, x)
                        t = time_fn(fwd, device, timing["warmup"], timing["runs"])
                        tracemalloc.start()
                        fwd()
                        tm_peak = tracemalloc.get_traced_memory()[1]
                        tracemalloc.stop()
                        mps_mb = torch.mps.current_allocated_memory() / 2**20 if device == "mps" else None
                        if mon:
                            mon.remove()
                        rows.append({**row, **_summary(t, bs), "status": "ok",
                                     "tracemalloc_peak_kb": tm_peak / 1024, "mps_alloc_mb": mps_mb})
                        log(f"{model_id} {device} {precision} bs={bs} {hooks}/{tap_set}: "
                            f"{rows[-1]['median_ms']:.3f} ms")
            if device == "mps":
                torch.mps.empty_cache()
    return pd.DataFrame(rows)


def measure_grad_steps(cfg: dict, gcfg: dict, timing: dict, log) -> pd.DataFrame:
    rows = []
    bs = gcfg["batch_size"]
    for model_id in cfg["models"]:
        for device in cfg["devices"]:
            model = build(model_id).to(device).train()
            opt = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
            x = torch.randn(bs, 3, 32, 32, device=device)
            y = torch.randint(0, MODELS[model_id].num_classes, (bs,), device=device)
            fwd_bwd = partial(_fwd_bwd, model, nn.CrossEntropyLoss(), x, y)
            train_step = partial(_train_step, fwd_bwd, opt)
            for kind, fn in (("fwd_bwd", fwd_bwd), ("train_step", train_step)):
                t = time_fn(fn, device, timing["warmup"], timing["runs"])
                rows.append({"model": model_id, "device": device, "precision": "fp32",
                             "batch": bs, "kind": kind, **_summary(t, bs)})
                log(f"{model_id} {device} {kind} bs={bs}: {rows[-1]['median_ms']:.1f} ms")
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- injection costs


def measure_injection(cfg: dict, icfg: dict, log) -> pd.DataFrame:
    """Per-injection fixed costs on CPU (the reporting device, §11.3)."""
    rows = []
    rng = np.random.default_rng(cfg["seed"])
    for model_id in cfg["models"]:
        fp32 = build(model_id).eval()
        int8 = quantize_ptq(fp32, [torch.randn(32, 3, 32, 32) for _ in range(2)])

        # FP32: flip + restore on a random weight tensor, budgets 1 and 16
        weights = [p for n, p in fp32.named_parameters() if n.endswith("weight") and p.dim() > 1]
        for count in (1, 16):
            t = time_fn(partial(_fp32_flip, rng, weights, count), "cpu", 5, icfg["runs"])
            rows.append({"model": model_id, "precision": "fp32", "kind": "flip_restore",
                         "count": count, "layer": "random", "median_ms": np.median(t) * 1e3})

        # INT8: unpack → XOR → repack, twice (flip and restore), per quantised layer
        qmods = {n: m for n, m in int8.named_modules()
                 if hasattr(m, "set_weight_bias") and callable(getattr(m, "weight", None))}
        for name, mod in qmods.items():
            numel = mod.weight().numel()
            for count in (1, 16):
                t = time_fn(partial(_int8_flip, rng, mod, numel, count), "cpu", 2,
                            max(10, icfg["runs"] // 10))
                rows.append({"model": model_id, "precision": "int8", "kind": "flip_restore",
                             "count": count, "layer": name, "layer_numel": numel,
                             "median_ms": np.median(t) * 1e3})

        for precision, model in (("fp32", fp32), ("int8", int8)):
            t = time_fn(lambda m=model: state_checksum(m), "cpu", 1, icfg["checksum_runs"])
            rows.append({"model": model_id, "precision": precision, "kind": "checksum",
                         "count": 0, "layer": "all", "median_ms": np.median(t) * 1e3})
        log(f"{model_id} injection costs measured ({len(qmods)} quantised layers)")
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- SVDD cost


def _synthetic_features(rng: np.random.Generator, kind: str, n: int, d: int) -> np.ndarray:
    """`gaussian` = N(0, I); `heavy` = Student-t(3) with spread scales (pessimistic, see README)."""
    if kind == "gaussian":
        return rng.standard_normal((n, d)).astype(np.float32)
    scales = np.logspace(-1, 1, d)
    return (rng.standard_t(3, size=(n, d)) * scales).astype(np.float32)


def _time_detector(det, x: np.ndarray, x_score: np.ndarray) -> tuple[float, float]:
    t0 = time.perf_counter()
    det.fit(x)
    fit_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    det.decision_function(x_score)
    return fit_s, len(x_score) / (time.perf_counter() - t0)


def measure_svdd(scfg: dict, ns: list[int], log) -> pd.DataFrame:
    """Fit/score cost of `OneClassSVM` and the Nystroem + `SGDOneClassSVM` path (SPEC §6.1)."""
    from sklearn.kernel_approximation import Nystroem
    from sklearn.linear_model import SGDOneClassSVM
    from sklearn.pipeline import make_pipeline
    from sklearn.svm import OneClassSVM

    rows = []
    rng = np.random.default_rng(0)
    for kind in scfg["data"]:
        for d in scfg["dims"]:
            x_score = _synthetic_features(rng, kind, scfg["score_n"], d)
            for n in ns:
                x = _synthetic_features(rng, kind, n, d)
                for nu in scfg["nus"]:
                    det = OneClassSVM(kernel="rbf", nu=nu, gamma="scale", cache_size=1000)
                    fit_s, score_rate = _time_detector(det, x, x_score)
                    rows.append({"method": "ocsvm_exact", "data": kind, "n": n, "d": d, "nu": nu,
                                 "fit_s": fit_s, "n_sv": len(det.support_),
                                 "score_per_s": score_rate})
                    log(f"OCSVM {kind} n={n} d={d} nu={nu}: fit {fit_s:.2f}s, "
                        f"{len(det.support_)} SV")
            for nu in scfg["nus"]:
                # same γ as OneClassSVM's gamma="scale", which Nystroem does not accept
                det = make_pipeline(Nystroem(gamma=1.0 / (d * x.var()), n_components=500,
                                             random_state=0),
                                    SGDOneClassSVM(nu=nu, random_state=0))
                fit_s, score_rate = _time_detector(det, x, x_score)
                rows.append({"method": "nystroem_sgd", "data": kind, "n": ns[-1], "d": d,
                             "nu": nu, "fit_s": fit_s, "n_sv": 500, "score_per_s": score_rate})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- provenance


def _git_commit() -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True)
        dirty = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True,
                               check=False)
        return out.stdout.strip() + ("-dirty" if dirty.stdout.strip() else "")
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def _sysctl(key: str) -> str:
    try:
        return subprocess.run(["sysctl", "-n", key], capture_output=True, text=True,
                              check=False).stdout.strip()
    except FileNotFoundError:
        return "unknown"


def sidecar(cfg: dict, quick: bool) -> dict:
    return {
        "milestone": "M-0",
        "quick": quick,
        "git_commit": _git_commit(),
        "config_hash": hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:16],
        "seed": cfg["seed"],
        "versions": {"python": platform.python_version(), "torch": torch.__version__,
                     "numpy": np.__version__, "sklearn": sklearn.__version__,
                     "pandas": pd.__version__},
        "machine": {"chip": _sysctl("machdep.cpu.brand_string"),
                    "memory_gb": int(_sysctl("hw.memsize") or 0) / 2**30,
                    "cpu_count": int(_sysctl("hw.ncpu") or 0),
                    "os": f"macOS {platform.mac_ver()[0]}"},
        "torch_threads": torch.get_num_threads(),
        "quantized_engine": BACKEND,
        "devices": cfg["devices"],
        "inputs": "synthetic N(0,1) 32x32x3; random-init weights; INT8 calibrated on random inputs",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }


def run(config_path: Path, out_dir: Path, quick: bool = False, only: set[str] | None = None) -> None:
    cfg = yaml.safe_load(config_path.read_text())
    mode = "quick" if quick else "full"
    if cfg.get("num_threads"):
        torch.set_num_threads(cfg["num_threads"])
    if "mps" in cfg["devices"] and not torch.backends.mps.is_available():
        cfg["devices"] = [d for d in cfg["devices"] if d != "mps"]
    _seed(cfg["seed"])
    warnings.filterwarnings("ignore", category=UserWarning)
    out_dir.mkdir(parents=True, exist_ok=True)

    def log(msg: str) -> None:
        print(f"[m0] {msg}", flush=True)

    parts = {
        "inference": lambda: measure_inference(cfg, cfg["timing"][mode], log),
        "grad": lambda: measure_grad_steps(cfg, cfg["grad_step"], cfg["grad_step"][mode], log),
        "injection": lambda: measure_injection(cfg, cfg["injection"][mode], log),
        "svdd": lambda: measure_svdd(cfg["svdd"], cfg["svdd"][mode]["n"], log),
    }
    for name, fn in parts.items():
        if only and name not in only:
            continue
        df = fn()
        df.to_parquet(out_dir / f"{name}.parquet", index=False)
        log(f"wrote {name}.parquet ({len(df)} rows)")
    (out_dir / "sidecar.json").write_text(json.dumps(sidecar(cfg, quick), indent=2))
