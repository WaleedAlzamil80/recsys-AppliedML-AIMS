import argparse
import os
from pathlib import Path

import numpy as np

from recsys_parallel.als_numba import fit_numba_als
from recsys_parallel.data import load_sparse_ratings
from visulizations.plots import plot_objective, plot_two_metrics


def main() -> None:
    project_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Train parallel ALS on MovieLens ratings.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=project_dir / "datasets" / "ml-32m",
        help="Directory containing ratings.csv.",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=min(8, os.cpu_count() or 1),
        help="Numba worker threads (default: up to 8).",
    )
    parser.add_argument("--epochs", type=int, default=30)
    args = parser.parse_args()

    data = load_sparse_ratings(args.data_dir, test_ratio=0.2, random_state=42)
    print(
        f"Loaded {data.train_by_user.nnz + data.test_by_user.nnz:,} ratings, "
        f"{len(data.user_ids):,} users, and {len(data.item_ids):,} movies; "
        f"training with {args.threads} thread(s)."
    )

    result = fit_numba_als(
        data,
        epochs=args.epochs,
        latent_dim=10,
        lamb=0.09,
        gamma=0.1,
        tau=0.9,
        random_state=42,
        threads=args.threads,
    )

    output_dir = project_dir / "output" / "practice_04"
    iterations = list(range(1, args.epochs + 1))
    plot_objective(
        iterations,
        result.losses,
        title="ALS Regularized Negative Log Likelihood",
        output_dir=output_dir / "plots",
        offset=0,
    )
    plot_objective(
        iterations,
        result.losses_test,
        title="ALS Regularized Negative Log Likelihood (Test)",
        output_dir=output_dir / "plots",
        offset=0,
    )
    plot_two_metrics(
        iterations,
        result.train_rmse,
        result.test_rmse,
        title="ALS Root Mean Squared Error",
        ylabel="RMSE",
        output_dir=output_dir / "plots",
        offset=0,
    )

    print(f"Final training objective: {result.losses[-1]:.4f}")
    print(f"Final training RMSE: {result.train_rmse[-1]:.4f}")
    print(f"Final test RMSE: {result.test_rmse[-1]:.4f}")
    print(f"Training objective non-increasing: {np.all(np.diff(result.losses) <= 1e-8)}")

    arrays_dir = output_dir / "npy_arrays"
    arrays_dir.mkdir(parents=True, exist_ok=True)
    np.save(arrays_dir / "user_biases.npy", result.user_biases)
    np.save(arrays_dir / "item_biases.npy", result.item_biases)
    np.save(arrays_dir / "user_vectors.npy", result.user_vectors)
    np.save(arrays_dir / "item_vectors.npy", result.item_vectors)
    np.save(arrays_dir / "user_ids.npy", data.user_ids)
    np.save(arrays_dir / "item_ids.npy", data.item_ids)


if __name__ == "__main__":
    main()
