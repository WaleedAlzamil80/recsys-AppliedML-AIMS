"""Practical 7: a mock recommender service and a simulated A/B test.

1. Train a baseline ALS model on 80% of the ratings and wrap it in a
   ``Recommender`` exposing ``createreco(userid, versionid)``.
   Version A ranks by ``u.v + b_n``; version B by ``u.v + 0.05 b_n``.
2. Simulate users: the "true" rating of a user for an item is the observed
   rating if it exists, else the prediction of an oracle model fitted on all
   ratings. A shown item is clicked with a position-discounted probability
   increasing in the true rating (see ``ClickModel``).
3. Randomly assign users to A or B, serve ``--sessions`` lists of
   ``--n-recs`` items, write the interaction log to CSV and run a Welch
   two-sample t-test on per-user click-through rates.
4. Sanity checks: an A/A test (false-positive rate should be ~5%) and a
   power analysis by re-randomising the user assignment many times.

Usage:
    python practice_07.py --data-dir datasets/ml-latest-small
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from recsys_models.als import ALSConfig, fit_als
from recsys_models.data import load_movielens, random_split
from recsys_serving import Recommender, createreco, set_default_recommender
from recsys_serving.abtest import (
    ClickModel, build_ground_truth, per_user_ctr, simulate_arm, welch_test)
from visulizations.plots import plot_bars, plot_histogram


def user_ctr_vector(log: pd.DataFrame, users: np.ndarray) -> np.ndarray:
    """CTR of each user id in ``users`` (aligned with ``users``)."""
    ctr = log.groupby("user_id")["click"].mean()
    return ctr.reindex(users).to_numpy()


def rerandomise(ctr_a: np.ndarray, ctr_b: np.ndarray, repeats: int,
                rng: np.random.Generator) -> np.ndarray:
    """p-values of Welch tests over random A/B assignments of all users."""
    n = len(ctr_a)
    p_values = np.empty(repeats)
    for r in range(repeats):
        in_a = rng.permutation(n) < n // 2
        a = np.where(in_a, ctr_a, ctr_b)[in_a]
        b = np.where(in_a, ctr_a, ctr_b)[~in_a]
        p_values[r] = stats.ttest_ind(b, a, equal_var=False).pvalue
    return p_values


def main() -> None:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path,
                        default=root / "datasets" / "ml-latest-small")
    parser.add_argument("--output-dir", type=Path,
                        default=root / "output" / "practice_07")
    parser.add_argument("--latent-dim", type=int, default=10)
    parser.add_argument("--oracle-dim", type=int, default=20)
    parser.add_argument("--lamb", type=float, default=0.09)
    parser.add_argument("--gamma", type=float, default=0.1)
    parser.add_argument("--tau", type=float, default=0.9)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--n-recs", type=int, default=10)
    parser.add_argument("--sessions", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    data = load_movielens(args.data_dir)
    train, _, test = random_split(data.ratings, data.n_users, data.n_items,
                                  fractions=(0.8, 0.0, 0.2), seed=args.seed)
    base = dict(lamb=args.lamb, gamma=args.gamma, tau=args.tau,
                epochs=args.epochs, seed=args.seed)
    model = fit_als(train, data.n_users, data.n_items,
                    ALSConfig(latent_dim=args.latent_dim, **base),
                    eval_sets={"test": test})
    print(f"service model: test RMSE {model.history['test_rmse'][-1]:.4f}")
    oracle = fit_als(data.ratings, data.n_users, data.n_items,
                     ALSConfig(latent_dim=args.oracle_dim, **base))
    truth = build_ground_truth(data, oracle, data.ratings)

    service = Recommender(data, model, train)
    set_default_recommender(service)
    demo_user = int(data.user_ids[0])
    demo = {v: createreco(demo_user, v, 5) for v in ("A", "B")}
    print("example createreco() output, user", demo_user)
    for version, recs in demo.items():
        print(f"  version {version}: {[title for _, title in recs]}")

    # --- potential outcomes of every user under every version ------------
    rng = np.random.default_rng(args.seed)
    click_model = ClickModel()
    users = np.arange(data.n_users)
    arms = {}
    for name, version in (("A", "A"), ("B", "B"), ("A2", "A")):
        arms[name] = simulate_arm(service, truth, version, users,
                                  click_model, args.n_recs, args.sessions,
                                  rng)
    user_ids = data.user_ids

    # --- the actual randomised experiment -----------------------------
    assignment = rng.permutation(data.n_users) < data.n_users // 2
    in_a = set(user_ids[assignment].tolist())
    log = pd.concat([
        arms["A"][arms["A"]["user_id"].isin(in_a)],
        arms["B"][~arms["B"]["user_id"].isin(in_a)],
    ], ignore_index=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    log.to_csv(args.output_dir / "ab_log.csv", index=False)
    test_ab = welch_test(per_user_ctr(log))
    print(f"A/B: CTR A {test_ab['ctr_control']:.4f} vs B "
          f"{test_ab['ctr_treatment']:.4f}, t={test_ab['t_statistic']:.2f}, "
          f"p={test_ab['p_value']:.4g}")

    # --- A/A sanity check on one randomisation ----------------------------
    aa_log = pd.concat([
        arms["A"][arms["A"]["user_id"].isin(in_a)],
        arms["A2"][~arms["A2"]["user_id"].isin(in_a)],
    ], ignore_index=True)
    aa_log["version"] = np.where(aa_log["user_id"].isin(in_a), "A", "B")
    test_aa = welch_test(per_user_ctr(aa_log))
    print(f"A/A: p={test_aa['p_value']:.4g}")

    # --- repeated randomisation: power and false-positive rate --------------
    ctr = {k: user_ctr_vector(v, user_ids) for k, v in arms.items()}
    p_ab = rerandomise(ctr["A"], ctr["B"], args.repeats, rng)
    p_aa = rerandomise(ctr["A"], ctr["A2"], args.repeats, rng)
    plots = args.output_dir / "plots"
    plot_histogram(p_ab, "p-value (A vs B)", "pvalues_ab.png", plots,
                   vline=0.05, vline_label="0.05")
    plot_histogram(p_aa, "p-value (A vs A)", "pvalues_aa.png", plots,
                   vline=0.05, vline_label="0.05")
    per_user = per_user_ctr(log)
    arm_ctr = [per_user.loc[per_user["version"] == v, "ctr"].to_numpy()
               for v in ("A", "B")]
    plot_bars(["A", "B"], [c.mean() for c in arm_ctr],
              [1.96 * c.std(ddof=1) / np.sqrt(len(c)) for c in arm_ctr],
              "mean per-user CTR", "ctr_comparison.png", plots)

    # --- behavioural differences between versions ------------------------
    popularity = np.bincount(train.items, minlength=data.n_items)
    item_index = {int(i): n for n, i in enumerate(data.item_ids)}
    behaviour = {}
    for name in ("A", "B"):
        shown = arms[name]["item_id"].map(item_index).to_numpy()
        users_idx = arms[name]["user_id"].map(
            {int(u): n for n, u in enumerate(user_ids)}).to_numpy()
        behaviour[name] = {
            "mean_log_popularity": float(np.mean(np.log1p(popularity[shown]))),
            "distinct_items": int(len(np.unique(shown))),
            "catalog_coverage": float(len(np.unique(shown)) / data.n_items),
            "mean_true_rating": float(np.mean(truth[users_idx, shown])),
            "ctr": float(arms[name]["click"].mean()),
        }
    overlap = []
    for user in users[:200]:
        a_items = {i for i, _ in createreco(int(user_ids[user]), "A", 10)}
        b_items = {i for i, _ in createreco(int(user_ids[user]), "B", 10)}
        overlap.append(len(a_items & b_items) / 10)

    result = {
        "dataset": args.data_dir.name,
        "service_model_test_rmse": model.history["test_rmse"][-1],
        "n_recs": args.n_recs, "sessions": args.sessions,
        "n_users": int(data.n_users),
        "n_users_a": int(assignment.sum()),
        "n_users_b": int((~assignment).sum()),
        "n_logged_interactions": int(len(log)),
        "click_model": {"slope": click_model.slope,
                        "centre": click_model.centre},
        "ab_test": test_ab, "aa_test": test_aa,
        "repeats": args.repeats,
        "power_ab": float(np.mean(p_ab < 0.05)),
        "false_positive_aa": float(np.mean(p_aa < 0.05)),
        "behaviour": behaviour,
        "mean_top10_overlap_a_b": float(np.mean(overlap)),
        "example_recs": {v: [t for _, t in r] for v, r in demo.items()},
        "example_user": demo_user,
    }
    (args.output_dir / "results.json").write_text(json.dumps(result, indent=2))
    print(f"power (A vs B, {args.repeats} re-randomisations): "
          f"{result['power_ab']:.3f}; A/A false-positive rate: "
          f"{result['false_positive_aa']:.3f}")


if __name__ == "__main__":
    main()
