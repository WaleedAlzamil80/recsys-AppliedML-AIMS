"""Alternating least squares for explicit ratings (NumPy implementation).

One implementation covers the three models of the report:

* biases only             (``latent_dim = 0``),
* biases + latent factors (``latent_dim = K``),
* the above with a genre side-information prior on the item vectors
  (``genre_matrix`` given), where item vectors are shrunk towards
  ``F_n^(-1/2) * sum_t f_t`` instead of towards zero.

All updates are exact block minimisers of the regularised negative
log-likelihood, so the training objective is non-increasing.

Notation follows the report: ``lamb`` is the observation precision
(lambda), ``tau`` the precision of the trait-vector prior, ``gamma`` the
precision of the bias prior and ``tau_feat`` the precision of the genre
vectors ``f_t``.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from recsys_models.data import Ratings, by_item, by_user


@dataclass
class ALSConfig:
    latent_dim: int = 10
    lamb: float = 0.09
    gamma: float = 0.1
    tau: float = 0.9
    tau_feat: float = 0.9
    epochs: int = 30
    seed: int = 42
    init_std: float = 0.1


@dataclass
class ALSModel:
    """Learned parameters and training history."""

    config: ALSConfig
    user_bias: np.ndarray
    item_bias: np.ndarray
    user_vecs: np.ndarray
    item_vecs: np.ndarray
    genre_vecs: np.ndarray | None = None
    item_scale: np.ndarray | None = None  # (N, T): G_nt / sqrt(F_n)
    history: dict[str, list[float]] = field(default_factory=dict)

    def predict(self, users: np.ndarray, items: np.ndarray) -> np.ndarray:
        """Predicted ratings for index pairs."""
        dots = np.einsum("rk,rk->r", self.user_vecs[users],
                         self.item_vecs[items])
        return dots + self.user_bias[users] + self.item_bias[items]

    def prior_mean(self) -> np.ndarray:
        """Item prior mean ``F_n^(-1/2) sum_t f_t`` (zeros without genres)."""
        if self.genre_vecs is None or self.item_scale is None:
            return np.zeros_like(self.item_vecs)
        return self.item_scale @ self.genre_vecs

    def rmse(self, data: Ratings) -> float:
        residual = data.values - self.predict(data.users, data.items)
        return float(np.sqrt(np.mean(residual ** 2)))


def item_scale_matrix(genre_matrix: np.ndarray) -> np.ndarray:
    """Return ``M`` with ``M_nt = G_nt / sqrt(F_n)`` (rows with no genre: 0)."""
    counts = genre_matrix.sum(axis=1, keepdims=True)
    safe = np.where(counts > 0, counts, 1.0)
    return genre_matrix / np.sqrt(safe)


def objective(model: ALSModel, train: Ratings) -> float:
    """Regularised negative log-likelihood (up to an additive constant)."""
    cfg = model.config
    residual = train.values - model.predict(train.users, train.items)
    value = 0.5 * cfg.lamb * float(np.sum(residual ** 2))
    value += 0.5 * cfg.gamma * float(
        np.sum(model.user_bias ** 2) + np.sum(model.item_bias ** 2))
    value += 0.5 * cfg.tau * float(np.sum(model.user_vecs ** 2))
    value += 0.5 * cfg.tau * float(
        np.sum((model.item_vecs - model.prior_mean()) ** 2))
    if model.genre_vecs is not None:
        value += 0.5 * cfg.tau_feat * float(np.sum(model.genre_vecs ** 2))
    return value


def _update_side(
    csr,
    bias: np.ndarray,
    vecs: np.ndarray,
    other_bias: np.ndarray,
    other_vecs: np.ndarray,
    cfg: ALSConfig,
    prior_mean: np.ndarray | None,
) -> None:
    """Update the bias and trait vector of every row of ``csr`` in place."""
    k = vecs.shape[1]
    ridge = cfg.tau * np.eye(k)
    for row in range(csr.shape[0]):
        start, stop = csr.indptr[row], csr.indptr[row + 1]
        count = stop - start
        if count == 0:
            bias[row] = 0.0
            if k:
                vecs[row] = 0.0 if prior_mean is None else prior_mean[row]
            continue
        cols = csr.indices[start:stop]
        values = csr.data[start:stop]
        other_b = other_bias[cols]
        partner = other_vecs[cols]
        fitted = other_b + (partner @ vecs[row] if k else 0.0)
        bias[row] = cfg.lamb * np.sum(values - fitted) / (
            cfg.lamb * count + cfg.gamma)
        if k:
            residual = values - bias[row] - other_b
            system = cfg.lamb * (partner.T @ partner) + ridge
            target = cfg.lamb * (partner.T @ residual)
            if prior_mean is not None:
                target = target + cfg.tau * prior_mean[row]
            vecs[row] = np.linalg.solve(system, target)


def _update_genre_vecs(
    genre_vecs: np.ndarray,
    item_vecs: np.ndarray,
    scale: np.ndarray,
    cfg: ALSConfig,
) -> None:
    """Exact joint update of all genre vectors.

    Because the prior mean ``mu = M F`` is linear in the genre matrix ``F``,
    minimising ``tau/2 ||V - M F||^2 + tau_f/2 ||F||^2`` over ``F`` is a
    ridge regression with the closed-form solution
    ``F = (tau M'M + tau_f I)^(-1) tau M'V`` (a ``T x T`` system).
    """
    gram = cfg.tau * (scale.T @ scale)
    gram += cfg.tau_feat * np.eye(scale.shape[1])
    genre_vecs[:] = np.linalg.solve(gram, cfg.tau * (scale.T @ item_vecs))


def fit_als(
    train: Ratings,
    n_users: int,
    n_items: int,
    cfg: ALSConfig,
    genre_matrix: np.ndarray | None = None,
    eval_sets: dict[str, Ratings] | None = None,
    verbose: bool = False,
) -> ALSModel:
    """Fit the model with ALS and record per-epoch objective and RMSE.

    Args:
        train: training ratings.
        n_users, n_items: matrix dimensions.
        cfg: hyper-parameters (see ``ALSConfig``).
        genre_matrix: optional ``(N, T)`` 0/1 genre matrix enabling the
            side-information prior.
        eval_sets: named held-out sets (e.g. ``{"val": ..., "test": ...}``);
            the RMSE on each is stored in ``history["<name>_rmse"]``.
        verbose: print one line per epoch.
    """
    rng = np.random.default_rng(cfg.seed)
    k = cfg.latent_dim
    model = ALSModel(
        config=cfg,
        user_bias=np.zeros(n_users),
        item_bias=np.zeros(n_items),
        user_vecs=rng.normal(0.0, cfg.init_std, (n_users, k)),
        item_vecs=rng.normal(0.0, cfg.init_std, (n_items, k)),
    )
    if genre_matrix is not None and k > 0:
        model.item_scale = item_scale_matrix(genre_matrix)
        model.genre_vecs = np.zeros((genre_matrix.shape[1], k))

    csr_user = by_user(train, n_users, n_items)
    csr_item = by_item(train, n_users, n_items)
    eval_sets = eval_sets or {}
    history: dict[str, list[float]] = {"objective": [], "train_rmse": []}
    for name in eval_sets:
        history[f"{name}_rmse"] = []

    for epoch in range(cfg.epochs):
        _update_side(csr_user, model.user_bias, model.user_vecs,
                     model.item_bias, model.item_vecs, cfg, None)
        _update_side(csr_item, model.item_bias, model.item_vecs,
                     model.user_bias, model.user_vecs, cfg,
                     model.prior_mean() if model.genre_vecs is not None
                     else None)
        if model.genre_vecs is not None:
            _update_genre_vecs(model.genre_vecs, model.item_vecs,
                               model.item_scale, cfg)
        history["objective"].append(objective(model, train))
        history["train_rmse"].append(model.rmse(train))
        for name, data in eval_sets.items():
            history[f"{name}_rmse"].append(model.rmse(data))
        if verbose:
            print(f"epoch {epoch + 1:3d}  objective "
                  f"{history['objective'][-1]:.2f}  train RMSE "
                  f"{history['train_rmse'][-1]:.4f}")
    model.history = history
    return model
