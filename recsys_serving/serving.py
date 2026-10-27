"""A minimal recommendation service with two model versions (A/B testing).

``createreco(userid, versionid)`` is the public entry point required by the
practical: it returns a list of ``(movie_id, title)`` tuples for a user,
produced by the model version the user is assigned to.

Version scores are ``u_m . v_n + w_version * b_n``. Version ``"A"`` is the
baseline (``w = 1``); version ``"B"`` down-weights the item bias
(``w = 0.05``) to favour personalisation over global popularity/quality.
"""
from __future__ import annotations

import numpy as np

from recsys_models.als import ALSModel
from recsys_models.data import MovieLens, Ratings

DEFAULT_VERSIONS: dict[str, float] = {"A": 1.0, "B": 0.05}


class Recommender:
    """Serve top-N lists from a trained ALS model."""

    def __init__(
        self,
        data: MovieLens,
        model: ALSModel,
        train: Ratings,
        versions: dict[str, float] | None = None,
    ) -> None:
        self.data = data
        self.model = model
        self.versions = dict(DEFAULT_VERSIONS if versions is None
                             else versions)
        self._user_index = {int(u): n for n, u in enumerate(data.user_ids)}
        self._item_ids = data.item_ids
        order = np.argsort(train.users, kind="stable")
        counts = np.bincount(train.users, minlength=data.n_users)
        self._seen_ptr = np.concatenate([[0], np.cumsum(counts)])
        self._seen_items = train.items[order]

    def scores(self, user_index: int | None, versionid: str) -> np.ndarray:
        """Score every item for a user index (``None``: unknown user)."""
        weight = self.versions[versionid]
        bias = weight * self.model.item_bias
        if user_index is None:
            return bias  # cold start: global item quality only
        return self.model.item_vecs @ self.model.user_vecs[user_index] + bias

    def seen(self, user_index: int) -> np.ndarray:
        return self._seen_items[self._seen_ptr[user_index]:
                                self._seen_ptr[user_index + 1]]

    def rank(self, user_index: int | None, versionid: str,
             n: int = 10, exclude: np.ndarray | None = None) -> np.ndarray:
        """Indices of the top ``n`` unseen items (optionally exclude more)."""
        scores = self.scores(user_index, versionid).copy()
        if user_index is not None:
            scores[self.seen(user_index)] = -np.inf
        if exclude is not None and len(exclude):
            scores[exclude] = -np.inf
        top = np.argpartition(-scores, n)[:n]
        return top[np.argsort(-scores[top])]

    def createreco(self, userid: int, versionid: str,
                   n: int = 10) -> list[tuple[int, str]]:
        """Top ``n`` recommendations as ``(movie_id, title)`` tuples."""
        if versionid not in self.versions:
            raise KeyError(f"unknown version {versionid!r}; "
                           f"choose from {sorted(self.versions)}")
        user_index = self._user_index.get(int(userid))
        top = self.rank(user_index, versionid, n)
        return [(int(self._item_ids[i]), self.data.titles[i]) for i in top]


_default: Recommender | None = None


def set_default_recommender(recommender: Recommender) -> None:
    """Register the recommender used by the module-level ``createreco``."""
    global _default
    _default = recommender


def createreco(userid: int, versionid: str,
               n: int = 10) -> list[tuple[int, str]]:
    """Module-level entry point: recommendations from the default service."""
    if _default is None:
        raise RuntimeError("call set_default_recommender() first")
    return _default.createreco(userid, versionid, n)
