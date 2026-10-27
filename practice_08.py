"""Practical 8: tuning lambda, tau, gamma and the latent dimension K.

Protocol: ratings are split 80/10/10 into train/validation/test. All choices
are made on validation RMSE; the test set is touched once, for the selected
configuration.

1. gamma sweep for the biases-only model (``K = 0``);
2. (lambda, tau) grid for ``K = 10`` (heat map);
3. K sweep over {2, 4, 8, 16, 32}; for every K the best tau of the grid is
   re-selected on validation, because larger K needs stronger regularisation.

Usage:
    python practice_08.py --data-dir datasets/ml-latest-small [--quick]
"""
import argparse
import json
from pathlib import Path

import numpy as np

from recsys_models.als import ALSConfig, fit_als
from recsys_models.data import load_movielens, random_split
from visulizations.plots import plot_heatmap, plot_lines

LAMBDAS = (0.03, 0.09, 0.3, 1.0)
TAUS = (0.1, 0.3, 0.9, 3.0, 9.0)
GAMMAS = (0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0)
LATENT_DIMS = (2, 4, 8, 16, 32)


def run(train, val, n_users, n_items, **kwargs) -> dict:
    """Fit once and summarise the final and best-epoch validation RMSE."""
    model = fit_als(train, n_users, n_items, ALSConfig(**kwargs),
                    eval_sets={"val": val})
    val_curve = model.history["val_rmse"]
    return {
        "train_rmse": model.history["train_rmse"][-1],
        "val_rmse": val_curve[-1],
        "best_epoch": int(np.argmin(val_curve)) + 1,
        "best_val_rmse": float(np.min(val_curve)),
        "monotone": bool(np.all(np.diff(model.history["objective"])
                                <= 1e-8)),
    }


def main() -> None:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path,
                        default=root / "datasets" / "ml-latest-small")
    parser.add_argument("--output-dir", type=Path,
                        default=root / "output" / "practice_08")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--base-dim", type=int, default=10,
                        help="K used for the (lambda, tau) grid")
    parser.add_argument("--gamma", type=float, default=0.1)
    parser.add_argument("--quick", action="store_true",
                        help="smaller grids for a fast smoke test")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    lambdas, taus = LAMBDAS, TAUS
    gammas, dims = GAMMAS, LATENT_DIMS
    if args.quick:
        lambdas, taus, gammas, dims = (0.09, 0.3), (0.3, 3.0), (0.1, 1.0), (
            2, 8)

    data = load_movielens(args.data_dir)
    train, val, test = random_split(data.ratings, data.n_users, data.n_items,
                                    seed=args.seed)
    common = dict(epochs=args.epochs, seed=args.seed)
    plots = args.output_dir / "plots"
    n_u, n_i = data.n_users, data.n_items

    # 1. bias regularisation (no latent factors) ---------------------------
    gamma_runs = []
    for gamma in gammas:
        out = run(train, val, n_u, n_i, latent_dim=0, lamb=1.0, gamma=gamma,
                  tau=1.0, **common)
        gamma_runs.append({"gamma": gamma, **out})
        print(f"[gamma {gamma:6.3f}] val RMSE {out['val_rmse']:.4f}")
    best_gamma = min(gamma_runs, key=lambda r: r["val_rmse"])

    # 2. (lambda, tau) grid for K = base_dim --------------------------------
    grid = np.zeros((len(lambdas), len(taus)))
    grid_runs = []
    for a, lamb in enumerate(lambdas):
        for b, tau in enumerate(taus):
            out = run(train, val, n_u, n_i, latent_dim=args.base_dim,
                      lamb=lamb, gamma=args.gamma, tau=tau, **common)
            grid[a, b] = out["val_rmse"]
            grid_runs.append({"lamb": lamb, "tau": tau, **out})
            print(f"[K={args.base_dim} lambda {lamb:5.2f} tau {tau:5.2f}] "
                  f"val {out['val_rmse']:.4f} train "
                  f"{out['train_rmse']:.4f}")
    a, b = np.unravel_index(np.argmin(grid), grid.shape)
    best_lamb, best_tau = lambdas[a], taus[b]
    plot_heatmap(grid, [f"{x:g}" for x in lambdas], [f"{x:g}" for x in taus],
                 r"prior precision $\tau$", r"noise precision $\lambda$",
                 f"Validation RMSE, K={args.base_dim}",
                 "lambda_tau_heatmap.png", plots, fmt=".2f")

    # 3. latent dimension: re-tune tau for each K --------------------------
    k_rows = []
    for k in dims:
        candidates = []
        for tau in taus:
            out = run(train, val, n_u, n_i, latent_dim=k, lamb=best_lamb,
                      gamma=args.gamma, tau=tau, **common)
            candidates.append({"tau": tau, **out})
        best = min(candidates, key=lambda r: r["val_rmse"])
        fixed = next(c for c in candidates if c["tau"] == best_tau)
        k_rows.append({"k": k, "best_tau": best["tau"],
                       "train_rmse": best["train_rmse"],
                       "val_rmse": best["val_rmse"],
                       "val_rmse_at_global_tau": fixed["val_rmse"],
                       "train_rmse_at_global_tau": fixed["train_rmse"],
                       "all_taus": candidates})
        print(f"[K={k:2d}] best tau {best['tau']:g} val "
              f"{best['val_rmse']:.4f} | val at tau={best_tau:g}: "
              f"{fixed['val_rmse']:.4f}")
    best_k_row = min(k_rows, key=lambda r: r["val_rmse"])
    ks = [r["k"] for r in k_rows]
    plot_lines(ks, {
        "train (tau re-tuned)": [r["train_rmse"] for r in k_rows],
        "validation (tau re-tuned)": [r["val_rmse"] for r in k_rows],
        f"validation (tau={best_tau:g} fixed)":
            [r["val_rmse_at_global_tau"] for r in k_rows],
    }, "latent dimension K", "RMSE", "RMSE versus latent dimension",
        "k_sweep.png", plots, logx=True)
    plot_lines([r["gamma"] for r in gamma_runs],
               {"validation": [r["val_rmse"] for r in gamma_runs],
                "train": [r["train_rmse"] for r in gamma_runs]},
               r"bias precision $\gamma$", "RMSE", "Biases-only model",
               "gamma_sweep.png", plots, logx=True)

    # 4. final evaluation on the untouched test set ---------------------------
    final_cfg = dict(latent_dim=best_k_row["k"], lamb=best_lamb,
                     gamma=args.gamma, tau=best_k_row["best_tau"], **common)
    final = fit_als(train, n_u, n_i, ALSConfig(**final_cfg),
                    eval_sets={"val": val, "test": test})
    bias_only = fit_als(
        train, n_u, n_i,
        ALSConfig(latent_dim=0, lamb=1.0, gamma=best_gamma["gamma"],
                  tau=1.0, **common), eval_sets={"val": val, "test": test})
    result = {
        "dataset": args.data_dir.name,
        "epochs": args.epochs,
        "split": {"train": len(train), "val": len(val), "test": len(test)},
        "gamma_runs": gamma_runs,
        "best_gamma": best_gamma["gamma"],
        "lambdas": list(lambdas), "taus": list(taus),
        "grid_val_rmse": grid.tolist(),
        "grid_runs": grid_runs,
        "best_lamb": best_lamb, "best_tau_grid": best_tau,
        "best_grid_val_rmse": float(grid[a, b]),
        "worst_grid_val_rmse": float(grid.max()),
        "k_rows": k_rows,
        "best_k": best_k_row["k"], "best_k_tau": best_k_row["best_tau"],
        "final": {
            "config": final_cfg,
            "train_rmse": final.history["train_rmse"][-1],
            "val_rmse": final.history["val_rmse"][-1],
            "test_rmse": final.history["test_rmse"][-1],
        },
        "biases_only": {
            "gamma": best_gamma["gamma"],
            "train_rmse": bias_only.history["train_rmse"][-1],
            "val_rmse": bias_only.history["val_rmse"][-1],
            "test_rmse": bias_only.history["test_rmse"][-1],
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "results.json").write_text(json.dumps(result, indent=2))
    print(f"\nSelected K={final_cfg['latent_dim']}, lambda={best_lamb:g}, "
          f"tau={final_cfg['tau']:g}: test RMSE "
          f"{result['final']['test_rmse']:.4f} (biases only "
          f"{result['biases_only']['test_rmse']:.4f})")


if __name__ == "__main__":
    main()
