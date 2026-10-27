"""Ranking metrics for top-k recommendation (implicit-feedback setting)."""
from __future__ import annotations

import numpy as np

from recsys_parallel.data import CompressedRatings


def ranking_metrics(
    scores: np.ndarray,
    train_pos: CompressedRatings,
    test_pos: CompressedRatings,
    ks: tuple[int, ...] = (5, 10, 20),
) -> dict[str, float]:
    """Mean Precision@k, Recall@k, NDCG@k and mean percentile rank.

    Args:
        scores: ``(M, N)`` score matrix, higher is better.
        train_pos: positives seen in training (excluded from the ranking).
        test_pos: held-out positives that should be retrieved.
        ks: cut-offs.

    Users without held-out positives are skipped. The mean percentile rank
    (MPR) is the average position of the held-out items in the ranking
    expressed as a fraction in ``[0, 1]`` (lower is better, 0.5 is random).
    """
    n_users, n_items = scores.shape
    max_k = max(ks)
    discounts = 1.0 / np.log2(np.arange(2, max_k + 2))
    sums = {f"{name}@{k}": 0.0 for name in ("precision", "recall", "ndcg")
            for k in ks}
    percentile_sum, percentile_count, evaluated = 0.0, 0, 0

    for user in range(n_users):
        t0, t1 = test_pos.indptr[user], test_pos.indptr[user + 1]
        if t1 == t0:
            continue
        relevant = test_pos.indices[t0:t1]
        row = scores[user].astype(np.float64, copy=True)
        seen = train_pos.indices[train_pos.indptr[user]:
                                 train_pos.indptr[user + 1]]
        row[seen] = -np.inf
        n_candidates = n_items - len(seen)

        top = np.argpartition(-row, max_k - 1)[:max_k]
        top = top[np.argsort(-row[top])]
        is_hit = np.isin(top, relevant)
        for k in ks:
            hits = int(is_hit[:k].sum())
            sums[f"precision@{k}"] += hits / k
            sums[f"recall@{k}"] += hits / len(relevant)
            dcg = float(np.sum(discounts[:k] * is_hit[:k]))
            ideal = float(np.sum(discounts[:min(k, len(relevant))]))
            sums[f"ndcg@{k}"] += dcg / ideal
        # Fraction of candidates scored strictly above each relevant item.
        above = (row[None, :] > row[relevant][:, None]).sum(axis=1)
        percentile_sum += float(np.sum(above / max(n_candidates - 1, 1)))
        percentile_count += len(relevant)
        evaluated += 1

    result = {key: value / evaluated for key, value in sums.items()}
    result["mpr"] = percentile_sum / percentile_count
    result["users_evaluated"] = float(evaluated)
    return result
