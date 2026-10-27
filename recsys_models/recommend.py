"""Qualitative tools: dummy users, top-N lists and polarisation analysis."""
from __future__ import annotations

import numpy as np
from scipy.stats import spearmanr

from recsys_models.als import ALSModel
from recsys_models.data import MovieLens, Ratings


def fold_in_user(
    model: ALSModel,
    item_indices: np.ndarray,
    values: np.ndarray,
    iterations: int = 25,
) -> tuple[np.ndarray, float]:
    """Fit a trait vector and bias for a new user with fixed item parameters.

    Runs the same alternating updates as the ALS user step on the few given
    ratings, while the item vectors and item biases stay fixed.
    """
    cfg = model.config
    k = cfg.latent_dim
    partner = model.item_vecs[item_indices]
    item_b = model.item_bias[item_indices]
    vec, bias = np.zeros(k), 0.0
    system = cfg.lamb * (partner.T @ partner) + cfg.tau * np.eye(k)
    for _ in range(iterations):
        bias = cfg.lamb * np.sum(values - item_b - partner @ vec) / (
            cfg.lamb * len(values) + cfg.gamma)
        vec = np.linalg.solve(
            system, cfg.lamb * (partner.T @ (values - bias - item_b)))
    return vec, float(bias)


def top_n(
    model: ALSModel,
    user_vec: np.ndarray,
    exclude: np.ndarray,
    n: int = 10,
    item_counts: np.ndarray | None = None,
    min_count: int = 0,
) -> np.ndarray:
    """Indices of the ``n`` best items for a trait vector.

    Items in ``exclude`` and items with fewer than ``min_count`` training
    ratings are never returned.
    """
    scores = model.item_vecs @ user_vec + model.item_bias
    scores[exclude] = -np.inf
    if item_counts is not None and min_count > 0:
        scores[item_counts < min_count] = -np.inf
    top = np.argpartition(-scores, n)[:n]
    return top[np.argsort(-scores[top])]


def polarization_table(
    data: MovieLens,
    model: ALSModel,
    train: Ratings,
    min_ratings: int = 50,
    n: int = 10,
) -> dict[str, object]:
    """Movies with the longest and shortest item vectors ``||v_n||``.

    Only movies with at least ``min_ratings`` training ratings are
    considered, because rarely rated movies are shrunk towards the origin by
    the prior. Also returns the Spearman correlation between ``||v_n||`` and
    the standard deviation of the observed ratings.
    """
    counts = np.bincount(train.items, minlength=data.n_items)
    sums = np.bincount(train.items, weights=train.values,
                       minlength=data.n_items)
    squares = np.bincount(train.items, weights=train.values ** 2,
                          minlength=data.n_items)
    mean = sums / np.maximum(counts, 1)
    std = np.sqrt(np.maximum(squares / np.maximum(counts, 1) - mean ** 2,
                             0.0))
    norms = np.linalg.norm(model.item_vecs, axis=1)
    eligible = np.flatnonzero(counts >= min_ratings)
    order = eligible[np.argsort(-norms[eligible])]

    def rows(indices):
        return [{"title": data.titles[i], "norm": float(norms[i]),
                 "count": int(counts[i]), "mean": float(mean[i]),
                 "std": float(std[i])} for i in indices]

    rho, p_value = spearmanr(norms[eligible], std[eligible])
    return {"most": rows(order[:n]), "least": rows(order[::-1][:n]),
            "spearman_norm_vs_std": float(rho),
            "spearman_p_value": float(p_value),
            "eligible_movies": int(len(eligible))}
