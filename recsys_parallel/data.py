import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class CompressedRatings:
    """A 2D ratings matrix stored as compressed rows using NumPy arrays.

    For a matrix with ``R`` rows and ``C`` columns containing ``K`` ratings:

    Attributes:
        indptr: ``int64`` array of shape ``(R + 1,)``. Row ``r`` occupies
            positions ``indptr[r]:indptr[r + 1]`` in ``indices`` and ``data``.
            The final value is ``K``; a row with no ratings has equal adjacent
            pointer values.
        indices: ``int32`` array of shape ``(K,)``. Column index for each
            stored rating, aligned position-by-position with ``data``.
        data: ``float64`` array of shape ``(K,)`` containing the rating values.
        shape: ``(R, C)`` matrix dimensions, including rows or columns with no
            stored ratings.
    """

    indptr: np.ndarray
    indices: np.ndarray
    data: np.ndarray
    shape: tuple[int, int]

    @property
    def nnz(self) -> int:
        """Number of stored ratings (the length of ``data``)."""
        return len(self.data)


@dataclass
class SparseRatings:
    """ID mappings and compressed train/test ratings used by Practice 4.

    ``user_ids[u]`` and ``item_ids[i]`` give the original IDs represented by
    zero-based matrix indices ``u`` and ``i``. Both ID arrays are sorted and
    have shape ``(number_of_users,)`` and ``(number_of_items,)`` respectively.

    Attributes:
        user_ids: Original user IDs, stored as a 1D NumPy array.
        item_ids: Original movie IDs, stored as a 1D NumPy array.
        train_by_user: Training ratings with shape ``(number_of_users,
            number_of_items)``; each row contains one user's rated item
            indices and rating values.
        train_by_item: The same training ratings transposed, with shape
            ``(number_of_items, number_of_users)``; each row contains one
            movie's rated user indices and rating values.
        test_by_user: Held-out ratings with shape ``(number_of_users,
            number_of_items)``, organized by user like ``train_by_user``.
    """

    user_ids: np.ndarray
    item_ids: np.ndarray
    train_by_user: CompressedRatings
    train_by_item: CompressedRatings
    test_by_user: CompressedRatings


def _compress_ratings(
    row_indices: np.ndarray,
    column_indices: np.ndarray,
    ratings: np.ndarray,
    shape: tuple[int, int],
) -> CompressedRatings:
    order = np.lexsort((column_indices, row_indices))
    sorted_rows = row_indices[order]
    counts = np.bincount(sorted_rows, minlength=shape[0])
    indptr = np.empty(shape[0] + 1, dtype=np.int64)
    indptr[0] = 0
    np.cumsum(counts, out=indptr[1:])
    return CompressedRatings(
        indptr=indptr,
        indices=column_indices[order].astype(np.int32, copy=False),
        data=ratings[order],
        shape=shape,
    )


def prepare_sparse_ratings(
    raw_user_ids: np.ndarray,
    raw_item_ids: np.ndarray,
    ratings: np.ndarray,
    test_ratio: float = 0.2,
    random_state: int = 42,
) -> SparseRatings:
    if not 0 <= test_ratio < 1:
        raise ValueError("test_ratio must be in the range [0, 1)")
    if not (len(raw_user_ids) == len(raw_item_ids) == len(ratings)):
        raise ValueError("user IDs, item IDs, and ratings must have equal lengths")

    user_ids, user_indices = np.unique(raw_user_ids, return_inverse=True)
    item_ids, item_indices = np.unique(raw_item_ids, return_inverse=True)
    user_indices = user_indices.astype(np.int32, copy=False)
    item_indices = item_indices.astype(np.int32, copy=False)
    rating_values = np.asarray(ratings, dtype=np.float64)

    rng = np.random.default_rng(random_state)
    is_test = rng.random(len(rating_values)) < test_ratio

    # Guarantee every entity has a training rating, even if random sampling holds out all its ratings.
    train_user_counts = np.bincount(user_indices[~is_test], minlength=len(user_ids))
    train_item_counts = np.bincount(item_indices[~is_test], minlength=len(item_ids))
    first_for_user = np.full(len(user_ids), len(rating_values), dtype=np.int64)
    first_for_item = np.full(len(item_ids), len(rating_values), dtype=np.int64)
    positions = np.arange(len(rating_values), dtype=np.int64)
    np.minimum.at(first_for_user, user_indices, positions)
    np.minimum.at(first_for_item, item_indices, positions)
    missing_users = np.flatnonzero(train_user_counts == 0)
    missing_items = np.flatnonzero(train_item_counts == 0)
    is_test[first_for_user[missing_users]] = False
    is_test[first_for_item[missing_items]] = False

    user_shape = (len(user_ids), len(item_ids))
    item_shape = (len(item_ids), len(user_ids))
    train_by_user = _compress_ratings(
        user_indices[~is_test],
        item_indices[~is_test],
        rating_values[~is_test],
        user_shape,
    )
    test_by_user = _compress_ratings(
        user_indices[is_test],
        item_indices[is_test],
        rating_values[is_test],
        user_shape,
    )
    train_by_item = _compress_ratings(
        item_indices[~is_test],
        user_indices[~is_test],
        rating_values[~is_test],
        item_shape,
    )

    return SparseRatings(
        user_ids=user_ids,
        item_ids=item_ids,
        train_by_user=train_by_user,
        train_by_item=train_by_item,
        test_by_user=test_by_user,
    )


def load_sparse_ratings(
    data_dir: Path,
    test_ratio: float = 0.2,
    random_state: int = 42,
) -> SparseRatings:
    ratings_path = data_dir / "ratings.csv"
    with ratings_path.open("r", encoding="utf-8", newline="") as source:
        rating_count = sum(1 for _ in source) - 1

    raw_user_ids = np.empty(rating_count, dtype=np.int32)
    raw_item_ids = np.empty(rating_count, dtype=np.int32)
    ratings = np.empty(rating_count, dtype=np.float64)
    with ratings_path.open("r", encoding="utf-8", newline="") as source:
        reader = csv.reader(source)
        next(reader, None)
        for index, row in enumerate(reader):
            raw_user_ids[index] = int(row[0])
            raw_item_ids[index] = int(row[1])
            ratings[index] = float(row[2])

    return prepare_sparse_ratings(
        raw_user_ids,
        raw_item_ids,
        ratings,
        test_ratio=test_ratio,
        random_state=random_state,
    )