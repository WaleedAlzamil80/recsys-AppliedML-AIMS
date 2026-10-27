"""Bayesian Personalized Ranking (BPR) for implicit feedback.

Positive feedback is derived from explicit ratings (rating >= threshold).
For a triplet ``(u, i, j)`` with ``i`` a positive and ``j`` a sampled
unobserved item, the model maximises ``ln sigmoid(x_uij)`` with
``x_uij = u_u . (v_i - v_j) + b_i - b_j`` and L2 regularisation, using
mini-batch stochastic gradient ascent.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from recsys_models.data import Ratings, to_csr
from recsys_parallel.data import CompressedRatings


@dataclass
class BPRConfig:
    latent_dim: int = 16
    learning_rate: float = 0.05
    reg: float = 0.01        # lambda_Theta for vectors
    reg_bias: float = 0.01   # lambda_Theta for item biases
    epochs: int = 60
    batch_size: int = 256
    init_std: float = 0.1
    use_item_bias: bool = True
    seed: int = 42


@dataclass
class BPRModel:
    config: BPRConfig
    user_vecs: np.ndarray
    item_vecs: np.ndarray
    item_bias: np.ndarray
    history: dict[str, list[float]] = field(default_factory=dict)

    def score_matrix(self) -> np.ndarray:
        """Scores ``u . v_i + b_i`` for every user/item pair, ``(M, N)``."""
        return self.user_vecs @ self.item_vecs.T + self.item_bias[None, :]


def split_positives(
    ratings: Ratings,
    n_users: int,
    threshold: float = 4.0,
    val_fraction: float = 0.1,
    test_fraction: float = 0.2,
    seed: int = 42,
) -> np.ndarray:
    """Hold out a fraction of every user's positives (rating >= threshold).

    Returns an ``int8`` label per rating: ``0`` training positive,
    ``1`` validation positive, ``2`` test positive and ``3`` not a positive.
    A user's last remaining training positive is never held out, so every
    user keeps at least one training interaction.
    """
    rng = np.random.default_rng(seed)
    positive = ratings.values >= threshold
    draw = rng.random(len(ratings))
    label = np.full(len(ratings), 3, dtype=np.int8)
    label[positive] = 0
    label[positive & (draw < test_fraction)] = 2
    label[positive & (draw >= test_fraction)
          & (draw < test_fraction + val_fraction)] = 1
    kept = np.bincount(ratings.users[label == 0], minlength=n_users)
    for user in np.flatnonzero(kept == 0):
        mine = np.flatnonzero((label != 0) & (label != 3)
                              & (ratings.users == user))
        if mine.size:
            label[mine[0]] = 0
    return label


def positives_of(ratings: Ratings, label: np.ndarray, which: int) -> Ratings:
    """The ``(user, item)`` pairs carrying a given label, with value 1."""
    chosen = label == which
    return Ratings(ratings.users[chosen], ratings.items[chosen],
                   np.ones(int(chosen.sum())))


def positives_csr(pos: Ratings, n_users: int, n_items: int):
    """Positives in compressed-row form (one row per user)."""
    return to_csr(pos.users, pos.items, pos.values, n_users, n_items)


def _sample_negatives(
    users: np.ndarray,
    sorted_keys: np.ndarray,
    n_items: int,
    rng: np.random.Generator,
    max_rounds: int = 20,
) -> np.ndarray:
    """Draw one uniform unobserved item per user, rejecting known positives."""
    negatives = rng.integers(0, n_items, size=len(users))
    user_keys = users.astype(np.int64) * n_items
    for _ in range(max_rounds):
        keys = user_keys + negatives
        slot = np.minimum(np.searchsorted(sorted_keys, keys),
                          len(sorted_keys) - 1)
        clash = sorted_keys[slot] == keys
        if not clash.any():
            break
        negatives[clash] = rng.integers(0, n_items, size=int(clash.sum()))
    return negatives


def bpr_loss(model: BPRModel, users, positives, negatives) -> float:
    """Mean negative log-likelihood ``-ln sigmoid(x_uij)`` of triplets."""
    u = model.user_vecs[users]
    x = np.einsum("bk,bk->b", u, model.item_vecs[positives]
                  - model.item_vecs[negatives])
    if model.config.use_item_bias:
        x = x + model.item_bias[positives] - model.item_bias[negatives]
    return float(np.mean(np.logaddexp(0.0, -x)))


def fit_bpr(
    train_pos: Ratings,
    n_users: int,
    n_items: int,
    cfg: BPRConfig,
    on_epoch: Callable[[int, BPRModel], dict[str, float]] | None = None,
    verbose: bool = False,
) -> BPRModel:
    """Train BPR with mini-batch SGD.

    Each epoch visits every training positive once, pairing it with a freshly
    sampled negative. Updates inside a mini-batch are accumulated with
    ``np.add.at`` (equivalent to sequential SGD when a row appears once).

    Args:
        on_epoch: optional callback ``f(epoch, model) -> dict`` whose values
            are appended to ``history`` (used for Precision@k curves).
    """
    rng = np.random.default_rng(cfg.seed)
    k = cfg.latent_dim
    model = BPRModel(
        config=cfg,
        user_vecs=rng.normal(0.0, cfg.init_std, (n_users, k)),
        item_vecs=rng.normal(0.0, cfg.init_std, (n_items, k)),
        item_bias=np.zeros(n_items),
    )
    users, positives = train_pos.users, train_pos.items
    sorted_keys = np.sort(users.astype(np.int64) * n_items + positives)

    # A fixed triplet sample gives a comparable loss curve across epochs.
    probe_neg = _sample_negatives(users, sorted_keys, n_items, rng)
    history: dict[str, list[float]] = {"bpr_loss": []}

    lr = cfg.learning_rate
    for epoch in range(cfg.epochs):
        negatives = _sample_negatives(users, sorted_keys, n_items, rng)
        order = rng.permutation(len(users))
        for start in range(0, len(order), cfg.batch_size):
            batch = order[start:start + cfg.batch_size]
            u, i, j = users[batch], positives[batch], negatives[batch]
            vec_u = model.user_vecs[u]
            vec_i = model.item_vecs[i]
            vec_j = model.item_vecs[j]
            x = np.einsum("bk,bk->b", vec_u, vec_i - vec_j)
            if cfg.use_item_bias:
                x = x + model.item_bias[i] - model.item_bias[j]
            # sigmoid(-x) = e^-x / (1 + e^-x), written in a stable form.
            weight = 0.5 * (1.0 - np.tanh(0.5 * x))
            w = weight[:, None]
            np.add.at(model.user_vecs, u,
                      lr * (w * (vec_i - vec_j) - cfg.reg * vec_u))
            np.add.at(model.item_vecs, i,
                      lr * (w * vec_u - cfg.reg * vec_i))
            np.add.at(model.item_vecs, j,
                      lr * (-w * vec_u - cfg.reg * vec_j))
            if cfg.use_item_bias:
                np.add.at(model.item_bias, i,
                          lr * (weight - cfg.reg_bias * model.item_bias[i]))
                np.add.at(model.item_bias, j,
                          lr * (-weight - cfg.reg_bias * model.item_bias[j]))
        history["bpr_loss"].append(bpr_loss(model, users, positives,
                                            probe_neg))
        if on_epoch is not None:
            for name, value in on_epoch(epoch, model).items():
                history.setdefault(name, []).append(value)
        if verbose:
            print(f"epoch {epoch + 1:3d}  BPR loss "
                  f"{history['bpr_loss'][-1]:.4f}")
    model.history = history
    return model


def popularity_scores(train_pos: CompressedRatings, n_users: int,
                      n_items: int) -> np.ndarray:
    """Non-personalised baseline: rank items by training positive count."""
    counts = np.bincount(train_pos.indices, minlength=n_items).astype(float)
    return np.tile(counts, (n_users, 1))
