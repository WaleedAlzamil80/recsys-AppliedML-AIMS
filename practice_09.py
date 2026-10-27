"""Practical 9: dummy-user recommendations and polarising movies.

* A "dummy user" who loves a few movies (default: the Lord of the Rings
  trilogy) is folded into a trained model: only the user's trait vector and
  bias are fitted, the item parameters stay fixed. We print the top-N
  recommended movies, with and without a minimum-popularity filter.
* Movies with the longest trait vectors ``||v_n||`` are the most polarising
  (strongly liked by some users, disliked by others); the shortest vectors
  are the most "neutral". We list both ends and correlate ``||v_n||`` with
  the standard deviation of the observed ratings.

By default the model is fitted here on all ratings of ``--data-dir``. To
analyse the 32M model of practical 4 instead, pass ``--arrays-dir
output/practice_04/npy_arrays`` together with its hyper-parameters.

Usage:
    python practice_09.py --data-dir datasets/ml-latest-small
"""
import argparse
import json
from pathlib import Path

import numpy as np

from recsys_models.als import ALSConfig, ALSModel, fit_als
from recsys_models.data import load_movielens
from recsys_models.recommend import fold_in_user, polarization_table, top_n
from visulizations.plots import plot_scatter

PROFILES = {
    "lord_of_the_rings": ["fellowship of the ring", "two towers",
                          "return of the king"],
    "animation": ["toy story (1995)", "finding nemo", "shrek (2001)",
                  "lion king"],
    "sci_fi": ["matrix, the (1999)", "blade runner (1982)",
               "star wars: episode iv", "alien (1979)"],
}


def pick_title(data, counts, pattern):
    """Most-rated movie whose title contains ``pattern`` (or ``None``)."""
    hits = data.find_titles(pattern)
    return max(hits, key=lambda i: counts[i]) if hits else None


def load_model_from_arrays(arrays_dir: Path, cfg: ALSConfig,
                           data) -> ALSModel:
    """Rebuild an ``ALSModel`` from the ``.npy`` files of practical 3/4."""
    item_ids = np.load(arrays_dir / "item_ids.npy")
    if not np.array_equal(item_ids, data.item_ids):
        raise ValueError("saved item ids do not match the dataset")
    return ALSModel(
        config=cfg,
        user_bias=np.load(arrays_dir / "user_biases.npy"),
        item_bias=np.load(arrays_dir / "item_biases.npy"),
        user_vecs=np.load(arrays_dir / "user_vectors.npy"),
        item_vecs=np.load(arrays_dir / "item_vectors.npy"),
    )


def main() -> None:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path,
                        default=root / "datasets" / "ml-latest-small")
    parser.add_argument("--output-dir", type=Path,
                        default=root / "output" / "practice_09")
    parser.add_argument("--tuning-results", type=Path,
                        default=root / "output" / "practice_08"
                        / "results.json",
                        help="use the configuration selected in practice 8")
    parser.add_argument("--arrays-dir", type=Path, default=None)
    parser.add_argument("--latent-dim", type=int, default=None)
    parser.add_argument("--lamb", type=float, default=None)
    parser.add_argument("--gamma", type=float, default=None)
    parser.add_argument("--tau", type=float, default=None)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--min-count", type=int, default=50,
                        help="popularity filter for the filtered list")
    parser.add_argument("--min-ratings", type=int, default=50,
                        help="minimum ratings for the polarisation table")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    chosen = dict(latent_dim=10, lamb=0.09, gamma=0.1, tau=0.9)
    if args.tuning_results.exists() and args.arrays_dir is None:
        final = json.loads(args.tuning_results.read_text())["final"]["config"]
        chosen.update({k: final[k] for k in chosen})
        print(f"using configuration selected in practice 8: {chosen}")
    for key in chosen:
        override = getattr(args, key)
        if override is not None:
            chosen[key] = override
    cfg = ALSConfig(epochs=args.epochs, seed=args.seed, **chosen)

    data = load_movielens(args.data_dir)
    if args.arrays_dir is not None:
        model = load_model_from_arrays(args.arrays_dir, cfg, data)
    else:
        model = fit_als(data.ratings, data.n_users, data.n_items, cfg)
    counts = np.bincount(data.ratings.items, minlength=data.n_items)

    profiles_out = {}
    for name, patterns in PROFILES.items():
        liked = [pick_title(data, counts, p) for p in patterns]
        liked = [i for i in liked if i is not None]
        if len(liked) < 2:
            print(f"profile {name}: titles not found, skipped")
            continue
        vec, bias = fold_in_user(model, np.array(liked),
                                 np.full(len(liked), 5.0))
        entry = {"liked": [data.titles[i] for i in liked], "user_bias": bias}
        for label, minimum in (("unfiltered", 0),
                               (f"min{args.min_count}", args.min_count)):
            top = top_n(model, vec, np.array(liked), 10, counts, minimum)
            scores = model.item_vecs[top] @ vec + model.item_bias[top]
            entry[label] = [
                {"title": data.titles[i], "score": float(s),
                 "n_ratings": int(counts[i])} for i, s in zip(top, scores)]
        profiles_out[name] = entry
        print(f"\n{name}: liked {entry['liked']}")
        for row in entry[f"min{args.min_count}"]:
            print(f"   {row['score']:.2f}  {row['title']} "
                  f"({row['n_ratings']} ratings)")

    polar = polarization_table(data, model, data.ratings, args.min_ratings)
    print(f"\nSpearman(||v||, rating std) = "
          f"{polar['spearman_norm_vs_std']:.3f}")
    print("most polarising:", [r["title"] for r in polar["most"][:5]])
    print("least polarising:", [r["title"] for r in polar["least"][:5]])

    norms = np.linalg.norm(model.item_vecs, axis=1)
    mask = counts >= args.min_ratings
    std = np.sqrt(np.maximum(
        np.bincount(data.ratings.items, weights=data.ratings.values ** 2,
                    minlength=data.n_items) / np.maximum(counts, 1)
        - (np.bincount(data.ratings.items, weights=data.ratings.values,
                       minlength=data.n_items) / np.maximum(counts, 1)) ** 2,
        0.0))
    plot_scatter(norms[mask], std[mask], r"item vector length $\|v_n\|$",
                 "std of observed ratings", "norm_vs_rating_std.png",
                 args.output_dir / "plots")

    result = {"dataset": args.data_dir.name, "config": chosen,
              "min_count": args.min_count,
              "min_ratings": args.min_ratings, "profiles": profiles_out,
              "polarization": polar}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "results.json").write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
