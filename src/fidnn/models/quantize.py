"""INT8 post-training static quantisation, FX + qnnpack, CPU-only (SPEC §3)."""

import copy
import warnings
from collections.abc import Iterable

import torch
from torch import nn
from torch.ao.quantization import get_default_qconfig_mapping
from torch.ao.quantization.quantize_fx import convert_fx, prepare_fx

SUPPORTED_ENGINES = torch.backends.quantized.supported_engines


def quantize_ptq(model: nn.Module, calib_batches: Iterable[torch.Tensor]) -> nn.Module:
    """INT8 copy of `model`. See `prepare_fx` / `convert_fx`; calibrate on `clean_fit` only."""
    
    if "x86" in SUPPORTED_ENGINES:
        BACKEND = "x86"
    elif "onednn" in SUPPORTED_ENGINES:
        BACKEND = "onednn"
    elif "qnnpack" in SUPPORTED_ENGINES:
        BACKEND = "qnnpack"
    else:
        print("ENGINEE: ", SUPPORTED_ENGINES)
        raise RuntimeError(
            f"No supported quantization backend found. "
            f"Available: {SUPPORTED_ENGINES}"
        )
    
    torch.backends.quantized.engine = BACKEND
    model = copy.deepcopy(model).cpu().eval()
    batches = list(calib_batches)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        prepared = prepare_fx(model, get_default_qconfig_mapping(BACKEND), example_inputs=(batches[0],))
        with torch.no_grad():
            for x in batches:
                prepared(x)
        return convert_fx(prepared).eval()
