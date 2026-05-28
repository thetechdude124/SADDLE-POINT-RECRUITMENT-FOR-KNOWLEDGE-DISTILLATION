"""Device, seeding, and small helpers used across the package."""

from __future__ import annotations

import os
import random
from typing import Optional, Union

import numpy as np
import torch


def get_device(prefer: Optional[str] = None) -> torch.device:
    """Return the best available torch device.

    Selection order (when ``prefer`` is ``None``): CUDA -> MPS (Apple Silicon)
    -> CPU. ``prefer`` may be one of ``"cuda"``, ``"mps"``, ``"cpu"`` to force
    a specific backend; the call falls back to CPU if the requested backend
    is unavailable.
    """

    def _is_mps_available() -> bool:
        return (
            getattr(torch.backends, "mps", None) is not None
            and torch.backends.mps.is_available()
        )

    if prefer is not None:
        prefer = prefer.lower()
        if prefer == "cuda" and torch.cuda.is_available():
            return torch.device("cuda")
        if prefer == "mps" and _is_mps_available():
            return torch.device("mps")
        if prefer == "cpu":
            return torch.device("cpu")

    if torch.cuda.is_available():
        return torch.device("cuda")
    if _is_mps_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch (CPU + CUDA) for reproducibility."""

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def to_device(
    obj: Union[torch.Tensor, torch.nn.Module, list, tuple, dict],
    device: torch.device,
):
    """Recursively move tensors / modules to ``device``."""

    if isinstance(obj, torch.Tensor):
        return obj.to(device)
    if isinstance(obj, torch.nn.Module):
        return obj.to(device)
    if isinstance(obj, (list, tuple)):
        moved = [to_device(x, device) for x in obj]
        return type(obj)(moved)
    if isinstance(obj, dict):
        return {k: to_device(v, device) for k, v in obj.items()}
    return obj
