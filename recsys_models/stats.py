"""Descriptive statistics and power-law fitting for rating data."""
from __future__ import annotations

import numpy as np


def fit_power_law(values: np.ndarray, min_tail: int = 30,
                  max_candidates: int = 120) -> dict[str, float]:
    """Fit ``P(k) ~ k^(-alpha)`` to the tail of integer degrees.

    Follows Clauset, Shalizi and Newman (2009): for each candidate ``x_min``
    the exponent is the maximum-likelihood estimate (discrete approximation)

        alpha = 1 + n / sum_i ln(x_i / (x_min - 1/2)),

    and ``x_min`` is chosen to minimise the Kolmogorov-Smirnov distance
    between the empirical and fitted tail distributions.

    Returns ``alpha``, ``x_min``, the KS distance ``ks`` and the tail size
    ``n_tail`` (``alpha`` is NaN if no tail with ``min_tail`` points exists).
    """
    x = np.sort(np.asarray(values, dtype=float))
    candidates = np.unique(x)
    if len(candidates) > max_candidates:
        pick = np.linspace(0, len(candidates) - 1, max_candidates)
        candidates = candidates[pick.astype(int)]
    best = {"alpha": float("nan"), "x_min": float("nan"),
            "ks": float("inf"), "n_tail": 0.0}
    for x_min in candidates:
        tail = x[x >= x_min]
        if len(tail) < min_tail or x_min < 1:
            continue
        shifted = x_min - 0.5
        alpha = 1.0 + len(tail) / np.sum(np.log(tail / shifted))
        empirical = 1.0 - np.arange(len(tail)) / len(tail)
        fitted = ((tail - 0.5) / shifted) ** (1.0 - alpha)
        ks = float(np.max(np.abs(empirical - fitted)))
        if ks < best["ks"]:
            best = {"alpha": float(alpha), "x_min": float(x_min), "ks": ks,
                    "n_tail": float(len(tail))}
    return best


def degree_summary(counts: np.ndarray) -> dict[str, float]:
    """Min, max, mean, median and Gini coefficient of a degree sequence."""
    c = np.sort(np.asarray(counts, dtype=float))
    n = len(c)
    gini = float((2 * np.sum((np.arange(1, n + 1)) * c) / (n * c.sum()))
                 - (n + 1) / n)
    return {"min": float(c[0]), "max": float(c[-1]), "mean": float(c.mean()),
            "median": float(np.median(c)), "gini": gini}
