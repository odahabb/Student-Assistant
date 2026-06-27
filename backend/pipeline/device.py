"""
backend/pipeline/device.py
Multimodal RAG Educational Assistant
Student: Omar Dahab — 23100704

Shared device-selection helper for the RAG pipeline.

Controlled by env var SA_DEVICE: "gpu" (default) | "cpu" | "npu".
Always falls back to "cpu" if the requested device isn't actually usable —
this module never hard-fails on missing hardware/drivers/optional packages.
"""

import os
import logging

log = logging.getLogger(__name__)


def get_torch_device() -> str:
    """
    Returns a torch device string: "cpu" or "xpu".

    Used by embedder.py, retriever.py, generator.py, and loader.py (BLIP/EasyOCR)
    for torch .to(device)/SentenceTransformer(device=...) calls.

    NPU is NOT returned here — NPU uses a completely separate OpenVINO code
    path (see should_use_npu() below), not torch.
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
    True only when SA_DEVICE=npu was explicitly requested. Callers (generator.py,
    embedder.py) are responsible for actually attempting the OpenVINO NPU path and
    falling back to get_torch_device()/cpu if it fails.
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
    EasyOCR unconditionally wraps its detector/recognizer models in
    torch.nn.DataParallel for any non-"cpu" device (detection.py, recognition.py)
    — a legacy assumption that the only non-CPU device is CUDA. DataParallel's
    scatter() calls torch._C._scatter, a CUDA-only C++ binding that does not exist
    for Intel's XPU backend, so EasyOCR on "xpu" crashes with
    AttributeError: module 'torch._C' has no attribute '_scatter'.

    There's only one Arc GPU here, so DataParallel's multi-GPU splitting isn't
    needed anyway — this replaces it process-wide with a thin passthrough that
    just runs the wrapped module directly. Safe because nothing else in this
    project uses torch.nn.DataParallel.
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
