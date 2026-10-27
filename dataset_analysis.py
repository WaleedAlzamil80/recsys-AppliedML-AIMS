"""Exploratory analysis: rating statistics and power-law tail fits.

Complements ``practice_01.py`` with complementary CDFs on log-log axes and
maximum-likelihood power-law exponents for the user and item degrees.

Usage:
    python dataset_analysis.py --data-dir datasets/ml-latest-small
"""
import argparse
import json
from pathlib import Path

import numpy as np

from recsys_models.data import load_movielens
from recsys_models.stats import degree_summary, fit_power_law
from visulizations.plots import plot_ccdf


def main() -> None:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path,
                        default=root / "datasets" / "ml-latest-small")
    parser.add_argument("--output-dir", type=Path,
                        default=root / "output" / "dataset_analysis")
    args = parser.parse_args()

    data = load_movielens(args.data_dir)
    ratings = data.ratings
    user_deg = np.bincount(ratings.users, minlength=data.n_users)
    item_deg = np.bincount(ratings.items, minlength=data.n_items)
    values, counts = np.unique(ratings.values, return_counts=True)

    fits = {"users": fit_power_law(user_deg), "items": fit_power_law(item_deg)}
    result = {
        "dataset": args.data_dir.name,
        "n_ratings": int(len(ratings)),
        "n_users": int(data.n_users),
        "n_items": int(data.n_items),
        "density": float(len(ratings) / (data.n_users * data.n_items)),
        "rating_mean": float(ratings.values.mean()),
        "rating_std": float(ratings.values.std()),
        "rating_histogram": {f"{v:.1f}": int(c)
                             for v, c in zip(values, counts)},
        "user_degree": degree_summary(user_deg),
        "item_degree": degree_summary(item_deg),
        "power_law": fits,
        "n_genres": len(data.genre_names),
    }
    plot_ccdf(
        {"users": user_deg, "items": item_deg},
        {k: (v["alpha"], v["x_min"]) for k, v in fits.items()},
        "degree_ccdf.png",
        args.output_dir,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "results.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
