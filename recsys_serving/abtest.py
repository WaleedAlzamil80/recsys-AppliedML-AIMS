"""Simulation and statistical analysis of an A/B test of two recommenders."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

from recsys_models.als import ALSModel
from recsys_models.data import MovieLens, Ratings
from recsys_serving.serving import Recommender


@dataclass
class ClickModel:
    """Click probability of a shown item, given the user's true rating.

    ``p(click) = examine(position) * sigmoid(slope * (rating - centre))``
    with ``examine(position) = 1 / log2(position + 2)`` (position from 0).
    """

    slope: float = 2.0
    centre: float = 4.0

    def probability(self, true_rating: np.ndarray,
                    position: np.ndarray) -> np.ndarray:
        like = 1.0 / (1.0 + np.exp(-self.slope * (true_rating - self.centre)))
        return like / np.log2(position + 2.0)


def build_ground_truth(
    data: MovieLens, oracle: ALSModel, all_ratings: Ratings,
) -> np.ndarray:
    """Dense ``(M, N)`` matrix of simulated true ratings.

    Observed ratings are used where they exist; elsewhere the prediction of
    an oracle model fitted on *all* ratings (a stand-in for real users'
    unobserved opinions), clipped to the rating scale.
    """
    if data.n_users * data.n_items > 80_000_000:
        raise MemoryError("dense ground truth is only for small datasets")
    truth = (oracle.user_vecs @ oracle.item_vecs.T
             + oracle.user_bias[:, None] + oracle.item_bias[None, :])
    truth = np.clip(truth, 0.5, 5.0)
    truth[all_ratings.users, all_ratings.items] = all_ratings.values
    return truth


def simulate_arm(
    service: Recommender,
    truth: np.ndarray,
    versionid: str,
    users: np.ndarray,
    click_model: ClickModel,
    n_recs: int,
    sessions: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Serve ``sessions`` lists of ``n_recs`` items per user and log clicks.

    In each session the user sees the best items not shown before, so a
    returning user is never shown the same title twice.
    """
    records = []
    for user in users:
        shown = np.empty(0, dtype=np.int64)
        for session in range(sessions):
            top = service.rank(int(user), versionid, n_recs, exclude=shown)
            position = np.arange(n_recs)
            probability = click_model.probability(truth[user, top], position)
            clicks = rng.random(n_recs) < probability
            for pos, (item, click) in enumerate(zip(top, clicks)):
                records.append((int(service.data.user_ids[user]),
                                int(service.data.item_ids[item]),
                                versionid, session, pos, int(click)))
            shown = np.concatenate([shown, top])
    return pd.DataFrame(records, columns=["user_id", "item_id", "version",
                                          "session", "position", "click"])


def per_user_ctr(log: pd.DataFrame) -> pd.DataFrame:
    """Click-through rate of every user (the unit of randomisation)."""
    grouped = log.groupby(["version", "user_id"])["click"].agg(
        ["sum", "count"])
    grouped["ctr"] = grouped["sum"] / grouped["count"]
    return grouped.reset_index()


def welch_test(ctr: pd.DataFrame, control: str = "A",
               treatment: str = "B") -> dict[str, float]:
    """Two-sample Welch t-test on per-user CTR (treatment minus control)."""
    a = ctr.loc[ctr["version"] == control, "ctr"].to_numpy()
    b = ctr.loc[ctr["version"] == treatment, "ctr"].to_numpy()
    test = stats.ttest_ind(b, a, equal_var=False)
    diff = float(b.mean() - a.mean())
    se = float(np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b)))
    pooled = np.sqrt(0.5 * (a.var(ddof=1) + b.var(ddof=1)))
    ci = test.confidence_interval(0.95)
    return {
        "n_control": float(len(a)), "n_treatment": float(len(b)),
        "ctr_control": float(a.mean()), "ctr_treatment": float(b.mean()),
        "difference": diff, "std_error": se,
        "ci95_low": float(ci.low), "ci95_high": float(ci.high),
        "t_statistic": float(test.statistic), "p_value": float(test.pvalue),
        "cohens_d": float(diff / pooled) if pooled > 0 else float("nan"),
        "relative_lift": float(diff / a.mean()) if a.mean() > 0
        else float("nan"),
    }
