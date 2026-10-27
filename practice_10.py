"""Practical 10 (bonus): variational Bayes instead of point estimates.

Fits a fully factorised Gaussian posterior over the user/item trait vectors
and both bias vectors by coordinate-ascent variational inference (CAVI) and
compares it with the MAP/ALS solution of the *same* model.

Scale invariance: the ALS objective only depends on the ratios tau/lambda and
gamma/lambda. To compare like with like we take the ratios selected in
practice 8 (if available) and set ``lambda`` to an absolute noise precision
(default 1.5, i.e. rating noise variance ~0.67) so that the posterior
predictive variance is meaningful.

Usage:
    python practice_10.py --data-dir datasets/ml-latest-small
"""
import argparse
import json
from pathlib import Path

import numpy as np

from recsys_models.als import ALSConfig, fit_als
from recsys_models.data import load_movielens, random_split
from recsys_models.vb import VBConfig, fit_vb
from visulizations.plots import plot_lines, plot_scatter

BUCKETS = [(1, 5), (6, 20), (21, 50), (51, 10 ** 9)]


def main() -> None:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path,
                        default=root / "datasets" / "ml-latest-small")
    parser.add_argument("--output-dir", type=Path,
                        default=root / "output" / "practice_10")
    parser.add_argument("--tuning-results", type=Path,
                        default=root / "output" / "practice_08"
                        / "results.json")
    parser.add_argument("--latent-dim", type=int, default=None)
    parser.add_argument("--lamb", type=float, default=1.5)
    parser.add_argument("--tau-ratio", type=float, default=None,
                        help="tau / lambda (default: from practice 8)")
    parser.add_argument("--gamma-ratio", type=float, default=None,
                        help="gamma / lambda (default: from practice 8)")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    latent_dim, tau_ratio, gamma_ratio = 10, 10.0, 1.0
    if args.tuning_results.exists():
        tuned = json.loads(args.tuning_results.read_text())["final"]["config"]
        latent_dim = tuned["latent_dim"]
        tau_ratio = tuned["tau"] / tuned["lamb"]
        gamma_ratio = tuned["gamma"] / tuned["lamb"]
        print(f"ratios from practice 8: tau/lambda={tau_ratio:.3g}, "
              f"gamma/lambda={gamma_ratio:.3g}, K={latent_dim}")
    latent_dim = args.latent_dim or latent_dim
    tau_ratio = args.tau_ratio or tau_ratio
    gamma_ratio = args.gamma_ratio or gamma_ratio
    lamb = args.lamb
    tau, gamma = tau_ratio * lamb, gamma_ratio * lamb

    data = load_movielens(args.data_dir)
    train, val, test = random_split(data.ratings, data.n_users, data.n_items,
                                    seed=args.seed)
    evals = {"val": val, "test": test}

    map_model = fit_als(
        train, data.n_users, data.n_items,
        ALSConfig(latent_dim=latent_dim, lamb=lamb, gamma=gamma, tau=tau,
                  epochs=args.epochs, seed=args.seed), eval_sets=evals)
    vb_model = fit_vb(
        train, data.n_users, data.n_items,
        VBConfig(latent_dim=latent_dim, lamb=lamb, gamma=gamma, tau=tau,
                 epochs=args.epochs, seed=args.seed), eval_sets=evals)

    elbo = np.array(vb_model.history["elbo"])
    vb_scores = vb_model.predictive_scores(test)
    # MAP has no posterior: use a constant noise variance (validation MSE).
    noise_var = map_model.history["val_rmse"][-1] ** 2
    err = test.values - map_model.predict(test.users, test.items)
    map_scores = {
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "log_density": float(np.mean(
            -0.5 * np.log(2 * np.pi * noise_var) - err ** 2 / (2 * noise_var)
        )),
        "coverage90": float(np.mean(np.abs(err) <= 1.6449
                                    * np.sqrt(noise_var))),
    }

    # RMSE of both models by number of training ratings of the item.
    item_counts = np.bincount(train.items, minlength=data.n_items)[test.items]
    vb_mean, vb_var = vb_model.predict(test.users, test.items)
    map_pred = map_model.predict(test.users, test.items)
    buckets = []
    for low, high in BUCKETS:
        mask = (item_counts >= low) & (item_counts <= high)
        if mask.any():
            buckets.append({
                "bin": f"{low}-{high}" if high < 10 ** 9 else f">={low}",
                "n_test": int(mask.sum()),
                "rmse_map": float(np.sqrt(np.mean(
                    (test.values[mask] - map_pred[mask]) ** 2))),
                "rmse_vb": float(np.sqrt(np.mean(
                    (test.values[mask] - vb_mean[mask]) ** 2))),
                "mean_pred_std": float(np.mean(np.sqrt(vb_var[mask])))})

    plots = args.output_dir / "plots"
    plot_lines(list(range(1, args.epochs + 1)),
               {"ELBO": elbo.tolist()}, "iteration", "ELBO",
               "Variational objective", "elbo.png", plots, markers=False)
    train_counts = np.bincount(train.items, minlength=data.n_items)
    plot_scatter(train_counts, np.sqrt(vb_model.item_bias_var),
                 "ratings of the movie (log scale)",
                 r"posterior std of item bias", "bias_uncertainty.png",
                 plots, logx=True)

    result = {
        "dataset": args.data_dir.name,
        "config": {"latent_dim": latent_dim, "lamb": lamb, "tau": tau,
                   "gamma": gamma, "tau_ratio": tau_ratio,
                   "gamma_ratio": gamma_ratio, "epochs": args.epochs},
        "elbo_first": float(elbo[0]), "elbo_last": float(elbo[-1]),
        "elbo_monotone": bool(np.all(np.diff(elbo) >= -1e-6)),
        "map": map_scores, "vb": vb_scores,
        "map_train_rmse": map_model.history["train_rmse"][-1],
        "vb_train_rmse": vb_model.history["train_rmse"][-1],
        "map_val_rmse": map_model.history["val_rmse"][-1],
        "vb_val_rmse": vb_model.history["val_rmse"][-1],
        "buckets": buckets,
        "item_bias_std_min": float(np.sqrt(vb_model.item_bias_var).min()),
        "item_bias_std_max": float(np.sqrt(vb_model.item_bias_var).max()),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "results.json").write_text(json.dumps(result, indent=2))
    print(f"MAP test RMSE {map_scores['rmse']:.4f} | VB test RMSE "
          f"{vb_scores['rmse']:.4f} | VB coverage90 "
          f"{vb_scores['coverage90']:.3f} | ELBO monotone "
          f"{result['elbo_monotone']}")


if __name__ == "__main__":
    main()
