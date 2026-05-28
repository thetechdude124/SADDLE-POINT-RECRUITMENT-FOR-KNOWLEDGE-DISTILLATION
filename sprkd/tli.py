"""Transfer Learning by Injection (TLI).

This module implements the *shape-aware tensor injection* primitive that lies
at the heart of TLI [Czyzewski 2020]:

.. code-block:: text

    fn_inject(src, dst):
        for each axis i:
            if src.shape[i] < dst.shape[i]: dst[(b-a)//2:..., ...] = src
            elif src.shape[i] > dst.shape[i]: dst = src[(a-b)//2:..., ...]
            else: dst = src

For matched-depth architectures (e.g. the Malaria teacher / student CNNs in
Experiment 1 of the SPRKD paper), pairing layers by ``state_dict`` key and
applying ``fn_inject`` element-wise is functionally equivalent to the full
graph-matching TLI of the original ``tli-pytorch`` package while avoiding the
heavy ``karateclub`` / ``timm`` / ``graphviz`` dependencies.

For cross-architecture injection (e.g. ResNet-101 -> ResNet-18 of Experiment
2) install the optional ``[tli-full]`` extras and use
:func:`sprkd.tli.transfer_via_graph`.

References
----------
* Czyzewski, M. *Neural Network Weight Initialization through Transfer
  Learning* (2020). https://arxiv.org/pdf/2006.12986.pdf
* Original implementation:
  https://github.com/maciejczyzewski/tli-pytorch (MIT license).
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Core primitive: shape-aware center-crop / center-pad injection
# ---------------------------------------------------------------------------

@torch.no_grad()
def fn_inject(from_tensor: torch.Tensor, to_tensor: torch.Tensor) -> None:
    """In-place center-aligned injection of ``from_tensor`` into ``to_tensor``.

    Each axis is handled independently:

    * ``from`` smaller than ``to``: ``from`` is centered inside ``to``.
    * ``from`` larger  than ``to``: a centered slice of ``from`` is copied.
    * Equal:                       a direct copy.

    This is the exact primitive used by the original ``tli-pytorch``
    implementation (Czyzewski 2020), reproduced here verbatim.
    """

    if from_tensor.shape == to_tensor.shape:
        to_tensor.copy_(from_tensor)
        return

    if from_tensor.dim() != to_tensor.dim():
        raise ValueError(
            f"fn_inject requires equal rank tensors; got {from_tensor.shape} "
            f"vs {to_tensor.shape}"
        )

    from_slices: List[slice] = []
    to_slices: List[slice] = []
    for a, b in zip(from_tensor.shape, to_tensor.shape):
        if a < b:
            lo = (b - a) // 2
            hi = b - (b - a + 1) // 2
            from_slices.append(slice(0, a))
            to_slices.append(slice(lo, hi))
        elif a > b:
            lo = (a - b) // 2
            hi = a - (a - b + 1) // 2
            from_slices.append(slice(lo, hi))
            to_slices.append(slice(0, b))
        else:
            from_slices.append(slice(0, a))
            to_slices.append(slice(0, b))

    to_tensor[tuple(to_slices)] = from_tensor[tuple(from_slices)].to(
        dtype=to_tensor.dtype, device=to_tensor.device
    )


# ---------------------------------------------------------------------------
# Layer pairing (matched-depth architectures)
# ---------------------------------------------------------------------------

def _named_param_dict(module: nn.Module) -> Dict[str, nn.Parameter]:
    return dict(module.named_parameters())


def pair_layers(
    student: nn.Module,
    teacher: nn.Module,
) -> List[Tuple[str, str]]:
    """Pair student and teacher parameters by matching ``state_dict`` keys.

    Falls back to positional pairing when keys disagree but counts match.

    Returns
    -------
    List[Tuple[str, str]]
        ``[(student_key, teacher_key), ...]`` pairs.
    """

    s_params = _named_param_dict(student)
    t_params = _named_param_dict(teacher)

    common = [k for k in s_params if k in t_params]
    if common:
        return [(k, k) for k in common]

    s_keys, t_keys = list(s_params), list(t_params)
    if len(s_keys) != len(t_keys):
        raise ValueError(
            "pair_layers cannot match: parameter counts differ "
            f"({len(s_keys)} vs {len(t_keys)}) and no shared keys exist. "
            "Use transfer_via_graph for heterogeneous architectures."
        )
    return list(zip(s_keys, t_keys))


@torch.no_grad()
def simple_inject(
    student: nn.Module,
    teacher: nn.Module,
    pairs: List[Tuple[str, str]] | None = None,
) -> List[Tuple[str, str]]:
    """Inject ``teacher`` parameters into ``student`` via center-crop/pad.

    For matched-depth architectures (Experiment 1 of the SPRKD paper) this is
    functionally equivalent to the full ``tli-pytorch`` ``transfer`` call but
    has no graph-matching dependencies.

    Parameters
    ----------
    student, teacher : nn.Module
        Models with paired layers (e.g. via :func:`pair_layers`).
    pairs : list of (str, str), optional
        Custom layer pairs. Defaults to :func:`pair_layers` output.

    Returns
    -------
    List[Tuple[str, str]]
        The pairs that were injected, useful for logging.
    """

    if pairs is None:
        pairs = pair_layers(student, teacher)

    s_params = _named_param_dict(student)
    t_params = _named_param_dict(teacher)

    for s_key, t_key in pairs:
        s_p = s_params[s_key]
        t_p = t_params[t_key]
        fn_inject(t_p.data, s_p.data)

    return pairs


@torch.no_grad()
def inject_state_list(
    student: nn.Module,
    state_tensors: List[torch.Tensor],
    teacher: nn.Module | None = None,
) -> None:
    """Inject a flat list of tensors (e.g. an averaged ASR) into ``student``.

    The list must align positionally with ``student.parameters()`` (this is
    the layout used by :class:`sprkd.saddle.SaddlePointRepository`).
    """

    student_params = list(student.parameters())
    if len(state_tensors) != len(student_params):
        raise ValueError(
            "inject_state_list expects one tensor per student parameter "
            f"({len(student_params)}); got {len(state_tensors)}"
        )

    if teacher is not None:
        # Match the legacy notebook flow: load tensors into the teacher first
        # (where shapes line up), then run shape-aware injection.
        teacher_params = list(teacher.parameters())
        if len(teacher_params) != len(state_tensors):
            raise ValueError(
                "inject_state_list with teacher expects matching parameter "
                f"counts ({len(teacher_params)}); got {len(state_tensors)}"
            )
        for tp, ts in zip(teacher_params, state_tensors):
            tp.data.copy_(ts.to(dtype=tp.dtype, device=tp.device))
        simple_inject(student, teacher)
        return

    for sp, ts in zip(student_params, state_tensors):
        fn_inject(ts.to(dtype=sp.dtype, device=sp.device), sp.data)


# ---------------------------------------------------------------------------
# Optional: full graph-matching TLI (heavy deps)
# ---------------------------------------------------------------------------

def transfer_via_graph(student: nn.Module, teacher: nn.Module, **kwargs):
    """Full TLI via the original ``tli-pytorch`` graph-matching algorithm.

    This requires the ``[tli-full]`` extras (``networkx``, ``graphviz``,
    ``karateclub``). Used in the paper for ResNet-101 -> ResNet-18 of
    Experiment 2.

    Raises
    ------
    ImportError
        If the ``[tli-full]`` extras are not installed.
    """

    try:
        from sprkd._tli_vendor import transfer  # type: ignore[attr-defined]
    except ImportError as e:  # pragma: no cover - optional path
        raise ImportError(
            "transfer_via_graph requires the 'tli-full' extras. "
            "Install with: pip install 'sprkd[tli-full]' "
            "and ensure sprkd/_tli_vendor.py is present."
        ) from e

    return transfer(teacher, student, inject=True, **kwargs)
