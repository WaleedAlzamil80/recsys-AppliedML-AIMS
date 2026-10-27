"""Fast MovieLens loading, random splits and compressed sparse helpers.

The loaders in ``recsys_data`` keep ratings as Python objects, which is easy
to read but slow for the hyper-parameter sweeps. This module keeps ratings as
parallel NumPy arrays (``users``, ``items``, ``values``) with zero-based
indices, and builds the CSR layout of ``recsys_parallel.data`` on demand.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from recsys_parallel.data import CompressedRatings

NO_GENRES = "(no genres listed)"


@dataclass
class Ratings:
    """Rating triples as parallel arrays with zero-based indices."""

    users: np.ndarray
    items: np.ndarray
    values: np.ndarray

    def __len__(self) -> int:
        return len(self.values)

    def subset(self, mask: np.ndarray) -> "Ratings":
        """Return the ratings selected by a boolean mask or index array."""
        return Ratings(self.users[mask], self.items[mask], self.values[mask])


@dataclass
class MovieLens:
    """A MovieLens dataset with dense zero-based user and item indices.

    Attributes:
        user_ids: original user id of each user index, shape ``(M,)``.
        item_ids: original movie id of each item index, shape ``(N,)``.
        ratings: all ratings as index triples.
        titles: title of each item index.
        genre_names: sorted list of the ``T`` genre names (without the
            "(no genres listed)" placeholder).
        genre_matrix: ``(N, T)`` 0/1 matrix, 1 if movie ``n`` has genre ``t``.
    """

    user_ids: np.ndarray
    item_ids: np.ndarray
    ratings: Ratings
    titles: list[str]
    genre_names: list[str]
    genre_matrix: np.ndarray

    @property
    def n_users(self) -> int:
        return len(self.user_ids)

    @property
    def n_items(self) -> int:
        return len(self.item_ids)

    def find_titles(self, pattern: str) -> list[int]:
        """Item indices whose title contains ``pattern`` (case-insensitive)."""
        needle = pattern.lower()
        return [n for n, t in enumerate(self.titles) if needle in t.lower()]


def load_movielens(data_dir: Path) -> MovieLens:
    """Load ``ratings.csv`` and ``movies.csv`` from a MovieLens directory."""
    data_dir = Path(data_dir)
    table = pd.read_csv(
        data_dir / "ratings.csv",
        usecols=["userId", "movieId", "rating"],
        dtype={"userId": np.int64, "movieId": np.int64,
               "rating": np.float64},
    )
    user_ids, user_idx = np.unique(table["userId"].to_numpy(),
                                   return_inverse=True)
    item_ids, item_idx = np.unique(table["movieId"].to_numpy(),
                                   return_inverse=True)
    ratings = Ratings(
        user_idx.astype(np.int32),
        item_idx.astype(np.int32),
        table["rating"].to_numpy(dtype=np.float64),
    )

    movies = pd.read_csv(data_dir / "movies.csv").set_index("movieId")
    titles, genre_lists = [], []
    for movie_id in item_ids:
        if movie_id in movies.index:
            titles.append(str(movies.at[movie_id, "title"]))
            genre_string = str(movies.at[movie_id, "genres"])
        else:
            titles.append(f"Unknown movie {movie_id}")
            genre_string = NO_GENRES
        names = [g for g in genre_string.split("|") if g != NO_GENRES]
        genre_lists.append(names)

    genre_names = sorted({g for names in genre_lists for g in names})
    position = {g: t for t, g in enumerate(genre_names)}
    genre_matrix = np.zeros((len(item_ids), len(genre_names)))
    for n, names in enumerate(genre_lists):
        for g in names:
            genre_matrix[n, position[g]] = 1.0
    return MovieLens(user_ids, item_ids, ratings, titles, genre_names,
                     genre_matrix)


def random_split(
    ratings: Ratings,
    n_users: int,
    n_items: int,
    fractions: tuple[float, float, float] = (0.8, 0.1, 0.1),
    seed: int = 42,
) -> tuple[Ratings, Ratings, Ratings]:
    """Randomly split ratings into train, validation and test sets.

    Every user and every item is guaranteed at least one training rating, so
    that all trait vectors and biases are estimated from data. Ratings are
    moved from validation/test to train when needed.
    """
    if not np.isclose(sum(fractions), 1.0):
        raise ValueError("fractions must sum to one")
    rng = np.random.default_rng(seed)
    draw = rng.random(len(ratings))
    edges = np.cumsum(fractions)[:2]
    label = np.searchsorted(edges, draw, side="right")  # 0, 1 or 2

    positions = np.arange(len(ratings))
    for index, size in ((ratings.users, n_users), (ratings.items, n_items)):
        in_train = np.bincount(index[label == 0], minlength=size) > 0
        first = np.full(size, len(ratings), dtype=np.int64)
        np.minimum.at(first, index, positions)
        label[first[~in_train]] = 0
    return tuple(ratings.subset(label == k) for k in range(3))  # type: ignore


def to_csr(
    rows: np.ndarray, cols: np.ndarray, values: np.ndarray, n_rows: int,
    n_cols: int,
) -> CompressedRatings:
    """Build the compressed-row layout used by ``recsys_parallel``."""
    order = np.lexsort((cols, rows))
    counts = np.bincount(rows, minlength=n_rows)
    indptr = np.zeros(n_rows + 1, dtype=np.int64)
    np.cumsum(counts, out=indptr[1:])
    return CompressedRatings(
        indptr=indptr,
        indices=cols[order].astype(np.int32, copy=False),
        data=np.asarray(values, dtype=np.float64)[order],
        shape=(n_rows, n_cols),
    )


def by_user(ratings: Ratings, n_users: int, n_items: int):
    """Ratings in compressed-row form, one row per user."""
    return to_csr(ratings.users, ratings.items, ratings.values, n_users,
                  n_items)


def by_item(ratings: Ratings, n_users: int, n_items: int):
    """Ratings in compressed-row form, one row per item."""
    return to_csr(ratings.items, ratings.users, ratings.values, n_items,
                  n_users)
