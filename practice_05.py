"""Practical 5: genre side information and 2D item embeddings.

Fits two ALS models with ``K = 2`` trait dimensions -- one with the standard
zero-mean prior on item vectors and one whose item prior is centred on the
normalised sum of the movie's genre vectors ``f_t`` -- and writes

* objective and RMSE curves for both,
* a scatter plot of the learned item vectors (similar/opposite movies),
* a plot of the learned genre vectors ``f_t``,
* a cold-start analysis (test RMSE by number of training ratings) using
  ``--cold-start-dim`` trait dimensions.

Usage:
    python practice_05.py --data-dir datasets/ml-latest-small
"""
import argparse
import json
from pathlib import Path

import numpy as np

from recsys_models.als import ALSConfig, fit_als
from recsys_models.data import load_movielens, random_split
from recsys_models.embedding import (
    anchor_annotations, cosine_neighbours, short_title, top_genre_groups)
from visulizations.plots import (
    plot_embedding_2d, plot_genre_vectors, plot_objective, plot_two_metrics)

COLD_START_BINS = [(1, 2), (3, 5), (6, 10), (11, 25), (26, 10 ** 9)]


def is_monotone(values: list[float], tol: float = 1e-8) -> bool:
    return bool(np.all(np.diff(values) <= tol))


def cold_start_table(data, train, test, plain, featured) -> list[dict]:
    """Test RMSE of both models grouped by item popularity in training."""
    counts = np.bincount(train.items, minlength=data.n_items)[test.items]
    rows = []
    for low, high in COLD_START_BINS:
        mask = (counts >= low) & (counts <= high)
        if not mask.any():
            continue
        entry = {"bin": f"{low}-{high}" if high < 10 ** 9 else f">={low}",
                 "n_test": int(mask.sum())}
        for name, model in (("plain", plain), ("genre", featured)):
            err = test.values[mask] - model.predict(test.users[mask],
                                                    test.items[mask])
            entry[f"rmse_{name}"] = float(np.sqrt(np.mean(err ** 2)))
        rows.append(entry)
    return rows


def main() -> None:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path,
                        default=root / "datasets" / "ml-latest-small")
    parser.add_argument("--output-dir", type=Path,
                        default=root / "output" / "practice_05")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--lamb", type=float, default=0.09)
    parser.add_argument("--gamma", type=float, default=0.1)
    parser.add_argument("--tau", type=float, default=0.9)
    parser.add_argument("--tau-feat", type=float, default=0.9)
    parser.add_argument("--cold-start-dim", type=int, default=10)
    parser.add_argument("--n-plot-items", type=int, default=400)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    data = load_movielens(args.data_dir)
    train, val, test = random_split(data.ratings, data.n_users, data.n_items,
                                    seed=args.seed)
    evals = {"val": val, "test": test}
    plots = args.output_dir / "plots"
    base = dict(lamb=args.lamb, gamma=args.gamma, tau=args.tau,
                tau_feat=args.tau_feat, epochs=args.epochs, seed=args.seed)
    iterations = list(range(1, args.epochs + 1))

    models = {}
    for name, genre in (("plain", None), ("genre", data.genre_matrix)):
        cfg = ALSConfig(latent_dim=2, **base)
        models[name] = fit_als(train, data.n_users, data.n_items, cfg, genre,
                               evals)
        hist = models[name].history
        plot_objective(iterations, hist["objective"],
                       title=f"Objective K2 {name}", output_dir=plots)
        plot_two_metrics(iterations, hist["train_rmse"], hist["val_rmse"],
                         title=f"RMSE K2 {name}", ylabel="RMSE",
                         output_dir=plots)
        print(f"[K=2 {name}] train {hist['train_rmse'][-1]:.4f} "
              f"val {hist['val_rmse'][-1]:.4f} "
              f"test {hist['test_rmse'][-1]:.4f} "
              f"monotone={is_monotone(hist['objective'])}")

    # --- 2D embeddings -------------------------------------------------
    counts = np.bincount(train.items, minlength=data.n_items)
    groups, group_names, selected = top_genre_groups(
        data, counts, n_items=args.n_plot_items)
    for name, title in (("plain", "Item vectors (no side information)"),
                        ("genre", "Item vectors (genre prior)")):
        coords = models[name].item_vecs[selected]
        notes = anchor_annotations(data, selected, coords)
        plot_embedding_2d(coords, groups, group_names, notes, title,
                          f"item_embedding_{name}.png", plots)
    plot_genre_vectors(models["genre"].genre_vecs, data.genre_names,
                       "Learned genre vectors $f_t$", "genre_vectors.png",
                       plots)

    # --- similar / opposite movies for anchor titles --------------------
    eligible = np.flatnonzero(counts >= 50)
    neighbours = {}
    for pattern in ("toy story (1995)", "pulp fiction", "fellowship of the"):
        hits = [i for i in data.find_titles(pattern) if i in set(eligible)]
        if not hits:
            continue
        anchor = max(hits, key=lambda i: counts[i])
        similar, opposite = cosine_neighbours(
            models["plain"].item_vecs, anchor, eligible)
        neighbours[short_title(data.titles[anchor], 40)] = {
            "similar": [short_title(data.titles[i], 40) for i in similar],
            "opposite": [short_title(data.titles[i], 40) for i in opposite]}

    # --- genre vector geometry -----------------------------------------
    f = models["genre"].genre_vecs
    genre_info = {
        name: {"x": float(f[t, 0]), "y": float(f[t, 1]),
               "norm": float(np.linalg.norm(f[t]))}
        for t, name in enumerate(data.genre_names)}

    # --- cold start with a larger K -------------------------------------
    cold_models = {}
    for name, genre in (("plain", None), ("genre", data.genre_matrix)):
        cfg = ALSConfig(latent_dim=args.cold_start_dim, **base)
        cold_models[name] = fit_als(train, data.n_users, data.n_items, cfg,
                                    genre, evals)
        print(f"[K={args.cold_start_dim} {name}] test RMSE "
              f"{cold_models[name].history['test_rmse'][-1]:.4f}")
    cold = cold_start_table(data, train, test, cold_models["plain"],
                            cold_models["genre"])

    result = {
        "dataset": args.data_dir.name,
        "hyperparameters": base,
        "k2": {name: {
            "train_rmse": m.history["train_rmse"][-1],
            "val_rmse": m.history["val_rmse"][-1],
            "test_rmse": m.history["test_rmse"][-1],
            "objective_first": m.history["objective"][0],
            "objective_last": m.history["objective"][-1],
            "objective_monotone": is_monotone(m.history["objective"]),
        } for name, m in models.items()},
        "cold_start_dim": args.cold_start_dim,
        "cold_start_full_model": {name: {
            "test_rmse": m.history["test_rmse"][-1],
            "objective_monotone": is_monotone(m.history["objective"]),
        } for name, m in cold_models.items()},
        "cold_start_buckets": cold,
        "genre_vectors": genre_info,
        "neighbours": neighbours,
        "embedding_groups": group_names,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "results.json").write_text(json.dumps(result, indent=2))
    print(f"Wrote results and plots to {args.output_dir}")


if __name__ == "__main__":
    main()
