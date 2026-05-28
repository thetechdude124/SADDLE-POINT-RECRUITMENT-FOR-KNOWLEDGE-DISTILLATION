"""Statistical tests used in the SPRKD paper (Table 1 + Section 5).

The headline numbers in the paper are backed by a paired McNemar test on the
SPRKD vs. RKD vs. Control-Student predictions over the held-out validation
split. This module provides:

* :func:`mcnemar_table` - build the 2x2 contingency table from two
  classifiers' predictions on the same inputs.
* :func:`mcnemar_test` - the discrete-binomial McNemar test (preferred for
  small ``b + c``, used in the paper) and the continuity-corrected
  chi-squared variant for larger samples.
* :func:`pairwise_mcnemar` - convenience runner over a dict of
  ``{name: predictions}`` against a shared ``targets`` tensor.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import comb
from typing import Dict, List, Optional, Sequence

import torch


@dataclass
class McNemarResult:
    """Outcome of a paired McNemar test."""

    n: int
    a: int  # both correct
    b: int  # model_a correct, model_b wrong
    c: int  # model_a wrong,   model_b correct
    d: int  # both wrong
    statistic: float
    p_value: float
    method: str
    accuracy_a: float = field(default=float("nan"))
    accuracy_b: float = field(default=float("nan"))

    def to_dict(self) -> dict:
        return {
            "n": self.n,
            "a_correct_b_correct": self.a,
            "a_correct_b_wrong":   self.b,
            "a_wrong_b_correct":   self.c,
            "a_wrong_b_wrong":     self.d,
            "statistic": self.statistic,
            "p_value": self.p_value,
            "method": self.method,
            "accuracy_a": self.accuracy_a,
            "accuracy_b": self.accuracy_b,
        }


def _as_long_tensor(x) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x.detach().to("cpu").long().reshape(-1)
    return torch.as_tensor(x, dtype=torch.long).reshape(-1)


def mcnemar_table(
    preds_a: torch.Tensor,
    preds_b: torch.Tensor,
    targets: torch.Tensor,
) -> tuple[int, int, int, int]:
    """Compute the (a, b, c, d) McNemar contingency table.

    a: ``A`` correct,  ``B`` correct
    b: ``A`` correct,  ``B`` wrong
    c: ``A`` wrong,    ``B`` correct
    d: ``A`` wrong,    ``B`` wrong
    """

    pa = _as_long_tensor(preds_a)
    pb = _as_long_tensor(preds_b)
    yt = _as_long_tensor(targets)
    if not (pa.numel() == pb.numel() == yt.numel()):
        raise ValueError(
            "preds_a, preds_b, and targets must have the same number of "
            f"elements; got {pa.numel()}, {pb.numel()}, {yt.numel()}."
        )
    ca = (pa == yt)
    cb = (pb == yt)
    a = int(( ca &  cb).sum().item())
    b = int(( ca & ~cb).sum().item())
    c = int((~ca &  cb).sum().item())
    d = int((~ca & ~cb).sum().item())
    return a, b, c, d


def _exact_binomial_p_value(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value (binomial distribution).

    For ``n = b + c`` discordant pairs the test statistic is ``min(b, c)``.
    Under H0, ``min(b, c) ~ Binomial(n, 0.5)``. We sum the two tails up to
    ``min(b, c)``.
    """

    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    pmf = [comb(n, i) * (0.5 ** n) for i in range(0, k + 1)]
    p = 2.0 * sum(pmf)
    return min(1.0, p)


def _chisq_p_value(stat: float) -> float:
    """Right-tail p-value for chi-squared with 1 DOF (no SciPy dependency).

    Uses the closed-form ``P(X^2_1 >= stat) = erfc(sqrt(stat / 2))``.
    """

    import math

    if stat <= 0:
        return 1.0
    return math.erfc(math.sqrt(stat / 2.0))


def mcnemar_test(
    preds_a: torch.Tensor,
    preds_b: torch.Tensor,
    targets: torch.Tensor,
    *,
    method: str = "exact",
    continuity: bool = True,
    use_statsmodels: bool = False,
) -> McNemarResult:
    """Run a paired McNemar test between two classifiers' predictions.

    Parameters
    ----------
    preds_a, preds_b : torch.Tensor
        Predicted class labels (long / int) for the two classifiers.
    targets : torch.Tensor
        Ground-truth labels.
    method : {"exact", "chi2"}
        ``"exact"`` (default, used in the paper) uses the discrete binomial
        distribution and is recommended for ``b + c < 25``. ``"chi2"`` uses
        the continuity-corrected chi-squared approximation.
    continuity : bool
        Whether to apply the Yates continuity correction in chi-squared mode.
    use_statsmodels : bool
        If ``True``, delegate to
        :func:`statsmodels.stats.contingency_tables.mcnemar` (the function
        used in the canonical
        ``EXPERIMENTAL_MODEL_EVALUATIONS.ipynb`` Colab). Falls back to the
        package's built-in implementation if ``statsmodels`` is not
        installed.
    """

    a, b, c, d = mcnemar_table(preds_a, preds_b, targets)
    n = a + b + c + d

    if use_statsmodels:
        try:
            from statsmodels.stats.contingency_tables import mcnemar as sm_mcnemar
        except ImportError:
            use_statsmodels = False  # silently fall back

    if use_statsmodels:
        contingency = [[d, c], [b, a]]   # statsmodels uses [[d c],[b a]] orientation
        result = sm_mcnemar(contingency, exact=(method == "exact"), correction=continuity)
        return McNemarResult(
            n=n,
            a=a, b=b, c=c, d=d,
            statistic=float(result.statistic),
            p_value=float(result.pvalue),
            method=f"statsmodels.{method}" + (" (corrected)" if continuity and method == "chi2" else ""),
            accuracy_a=(a + b) / max(1, n),
            accuracy_b=(a + c) / max(1, n),
        )

    if method == "exact":
        statistic = float(min(b, c))
        p_value = _exact_binomial_p_value(b, c)
        method_used = "exact (binomial)"
    elif method == "chi2":
        if (b + c) == 0:
            statistic = 0.0
            p_value = 1.0
        else:
            num = abs(b - c) - (1.0 if continuity else 0.0)
            statistic = float(max(0.0, num) ** 2 / (b + c))
            p_value = _chisq_p_value(statistic)
        method_used = "chi-squared" + (" (continuity-corrected)" if continuity else "")
    else:
        raise ValueError(f"unknown method: {method!r}; expected 'exact' or 'chi2'")

    return McNemarResult(
        n=n,
        a=a, b=b, c=c, d=d,
        statistic=statistic,
        p_value=p_value,
        method=method_used,
        accuracy_a=(a + b) / max(1, n),
        accuracy_b=(a + c) / max(1, n),
    )


def pairwise_mcnemar(
    predictions: Dict[str, torch.Tensor],
    targets: torch.Tensor,
    *,
    method: str = "exact",
) -> List[Dict]:
    """Run McNemar between every ordered pair of classifiers in ``predictions``.

    Returns a list of dicts (one per pair) with keys ``model_a``, ``model_b``,
    ``statistic``, ``p_value``, plus the underlying contingency cells. Useful
    for reproducing the full p-value matrix referenced in Table 1.
    """

    names = list(predictions)
    out = []
    for i, ai in enumerate(names):
        for j, bj in enumerate(names):
            if i >= j:
                continue
            res = mcnemar_test(predictions[ai], predictions[bj], targets, method=method)
            out.append({"model_a": ai, "model_b": bj, **res.to_dict()})
    return out
