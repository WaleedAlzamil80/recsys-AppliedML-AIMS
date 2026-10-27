"""Helpers to inspect and label learned item embeddings."""
from __future__ import annotations

import numpy as np

from recsys_models.data import MovieLens

# Well-known titles used as anchors when labelling embedding plots. Patterns
# are matched as lower-case substrings of the MovieLens title.
ANCHOR_PATTERNS = [
    "toy story (1995)", "pulp fiction", "fellowship of the ring",
    "star wars: episode iv", "matrix, the (1999)", "shawshank",
    "forrest gump", "titanic (1997)", "dumb & dumber", "silence of the lambs",
    "fargo", "jurassic park", "lion king", "terminator 2", "schindler",
    "braveheart", "godfather, the (1972)", "twister", "scream (1996)",
    "alien (1979)", "casablanca", "wallace & gromit",
]


def top_genre_groups(
    data: MovieLens, item_counts: np.ndarray, n_items: int = 400,
    n_genres: int = 7,
) -> tuple[np.ndarray, list[str], np.ndarray]:
    """Colour groups for the ``n_items`` most-rated movies.

    Each movie gets the index of its first genre among the ``n_genres`` most
    frequent genres of that subset (``-1`` if none). Returns
    ``(group per selected movie, group names, selected item indices)``.
    """
    selected = np.argsort(-item_counts)[:n_items]
    genre_freq = data.genre_matrix[selected].sum(axis=0)
    top = np.argsort(-genre_freq)[:n_genres]
    names = [data.genre_names[t] for t in top]
    groups = np.full(len(selected), -1, dtype=int)
    for row, item in enumerate(selected):
        for rank, t in enumerate(top):
            if data.genre_matrix[item, t] > 0:
                groups[row] = rank
                break
    return groups, names, selected


def anchor_annotations(
    data: MovieLens, selected: np.ndarray, extra_far: np.ndarray,
    n_far: int = 2,
) -> list[tuple[int, str]]:
    """``(row in selected, short label)`` for anchors and extreme movies."""
    position = {int(item): row for row, item in enumerate(selected)}
    chosen: dict[int, str] = {}
    for pattern in ANCHOR_PATTERNS:
        for item in data.find_titles(pattern):
            if item in position:
                chosen[position[item]] = short_title(data.titles[item])
                break
    norms = np.linalg.norm(extra_far, axis=1)
    for row in np.argsort(-norms)[:n_far]:
        chosen.setdefault(int(row), short_title(data.titles[selected[row]]))
    return sorted(chosen.items())


def short_title(title: str, width: int = 24) -> str:
    """Trim the release year and long names for plot labels."""
    base = title.rsplit(" (", 1)[0]
    return base if len(base) <= width else base[: width - 1] + "."


def cosine_neighbours(
    vectors: np.ndarray, anchor: int, eligible: np.ndarray, n: int = 3,
) -> tuple[list[int], list[int]]:
    """Most similar and most opposite eligible items to ``anchor``."""
    unit = vectors / np.maximum(np.linalg.norm(vectors, axis=1,
                                               keepdims=True), 1e-12)
    similarity = unit @ unit[anchor]
    candidates = eligible[eligible != anchor]
    order = candidates[np.argsort(-similarity[candidates])]
    return list(order[:n]), list(order[::-1][:n])
