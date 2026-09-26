"""
backend/pipeline/device.py
Multimodal RAG Educational Assistant
Student: Omar Dahab — 23100704

Shared device selection for the RAG pipeline.

Controlled by the SA_DEVICE env var: "gpu" (default) | "cpu" | "npu".
Falls back to "cpu" when the requested device is not usable; nothing here
raises on missing hardware, drivers or optional packages.
"""

import os
import logging

log = logging.getLogger(__name__)


def get_torch_device() -> str:
    """
    Return a torch device string: "cpu" or "xpu".

    Used by embedder.py, retriever.py, generator.py and loader.py (BLIP and
    EasyOCR) for .to(device) and SentenceTransformer(device=...) calls.

    "npu" is never returned: the NPU runs through OpenVINO rather than torch
    (see should_use_npu below).
    """
    requested = os.environ.get("SA_DEVICE", "gpu").lower()
    if requested != "gpu":
        return "cpu"
    try:
        import torch
        if hasattr(torch, "xpu") and torch.xpu.is_available():
            return "xpu"
        log.warning(
            "SA_DEVICE=gpu requested but torch.xpu.is_available() is False "
            "(CPU-only torch build or no Arc driver) — falling back to cpu"
        )
    except Exception as e:
        log.warning(f"GPU device check failed ({e}) — falling back to cpu")
    return "cpu"


def should_use_npu() -> bool:
    """
    True when SA_DEVICE=npu. Callers (generator.py, embedder.py) attempt the
    OpenVINO NPU path themselves and fall back to get_torch_device() if it
    fails.
    """
    return os.environ.get("SA_DEVICE", "gpu").lower() == "npu"


def get_easyocr_device():
    """EasyOCR's Reader(gpu=...) accepts True/False or a device string directly."""
    device = "xpu" if get_torch_device() == "xpu" else False
    if device == "xpu":
        patch_dataparallel_for_xpu()
    return device


def patch_dataparallel_for_xpu():
    """
    Replace torch.nn.DataParallel process-wide with a passthrough wrapper.

    EasyOCR wraps its detector and recogniser in DataParallel for any non-CPU
    device. DataParallel's scatter() calls torch._C._scatter, which exists
    only for CUDA, so on "xpu" EasyOCR raises AttributeError: module
    'torch._C' has no attribute '_scatter'. The replacement runs the wrapped
    module directly and does no device splitting. Applied once per process;
    nothing else in the project uses DataParallel.
    """
    import torch.nn as nn

    if getattr(nn.DataParallel, "_sa_xpu_patched", False):
        return

    class _PassthroughDataParallel(nn.Module):
        def __init__(self, module, *args, **kwargs):
            super().__init__()
            self.module = module

        def forward(self, *args, **kwargs):
            return self.module(*args, **kwargs)

    _PassthroughDataParallel._sa_xpu_patched = True
    nn.DataParallel = _PassthroughDataParallel
    log.info("Patched torch.nn.DataParallel -> passthrough for XPU (no multi-GPU scatter)")
