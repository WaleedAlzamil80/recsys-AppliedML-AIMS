"""Benchmark the parallel (Numba) ALS: time per epoch versus thread count.

Reports seconds per epoch, speed-up and parallel efficiency relative to one
thread, and (optionally) how much faster the Numba implementation is than
the NumPy one of ``recsys_models.als`` on a small dataset.

Usage:
    python benchmark_scaling.py --data-dir datasets/ml-32m \
        --threads 1 2 4 8 --epochs 2 --small-dir datasets/ml-latest-small
"""
import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

from recsys_models.als import ALSConfig, fit_als
from recsys_models.data import load_movielens, random_split
from recsys_parallel.als_numba import fit_numba_als
from recsys_parallel.data import load_sparse_ratings, prepare_sparse_ratings
from visulizations.plots import plot_lines

HYPER = dict(latent_dim=10, lamb=0.09, gamma=0.1, tau=0.9)


def warm_up() -> None:
    """Trigger Numba compilation on a tiny random problem."""
    rng = np.random.default_rng(0)
    users = rng.integers(0, 50, 2000)
    items = rng.integers(0, 80, 2000)
    values = rng.integers(1, 11, 2000) / 2.0
    tiny = prepare_sparse_ratings(users, items, values, 0.2, 0)
    fit_numba_als(tiny, epochs=1, random_state=0, threads=1, **HYPER)


def time_numba(data, threads: int, epochs: int) -> float:
    start = time.perf_counter()
    fit_numba_als(data, epochs=epochs, random_state=42, threads=threads,
                  **HYPER)
    return (time.perf_counter() - start) / epochs


def main() -> None:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path,
                        default=root / "datasets" / "ml-32m")
    parser.add_argument("--small-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path,
                        default=root / "output" / "benchmark")
    parser.add_argument("--threads", type=int, nargs="+",
                        default=[1, 2, 4, 8])
    parser.add_argument("--epochs", type=int, default=2)
    args = parser.parse_args()

    max_threads = os.cpu_count() or 1
    threads = [t for t in args.threads if t <= max_threads]
    warm_up()
    data = load_sparse_ratings(args.data_dir)
    n_ratings = data.train_by_user.nnz + data.test_by_user.nnz
    print(f"{n_ratings:,} ratings, {len(data.user_ids):,} users, "
          f"{len(data.item_ids):,} movies")

    seconds = {}
    for count in threads:
        seconds[count] = time_numba(data, count, args.epochs)
        print(f"{count:2d} thread(s): {seconds[count]:.2f} s/epoch")
    base = seconds[threads[0]]
    speedup = {t: base / s for t, s in seconds.items()}
    efficiency = {t: speedup[t] * threads[0] / t for t in threads}

    result = {
        "dataset": args.data_dir.name,
        "n_ratings": int(n_ratings),
        "n_users": int(len(data.user_ids)),
        "n_items": int(len(data.item_ids)),
        "n_train": int(data.train_by_user.nnz),
        "cpu_count": max_threads,
        "epochs_timed": args.epochs,
        "seconds_per_epoch": {str(t): s for t, s in seconds.items()},
        "speedup": {str(t): v for t, v in speedup.items()},
        "efficiency": {str(t): v for t, v in efficiency.items()},
        "ratings_per_second": {
            str(t): data.train_by_user.nnz / s for t, s in seconds.items()},
    }

    if args.small_dir is not None:
        small = load_movielens(args.small_dir)
        train, _, _ = random_split(small.ratings, small.n_users,
                                   small.n_items, (0.8, 0.0, 0.2))
        start = time.perf_counter()
        fit_als(train, small.n_users, small.n_items,
                ALSConfig(epochs=1, seed=42, **HYPER))
        numpy_time = time.perf_counter() - start
        sparse = load_sparse_ratings(args.small_dir)
        numba_time = time_numba(sparse, threads[-1], 1)
        result["small_comparison"] = {
            "dataset": args.small_dir.name,
            "numpy_seconds_per_epoch": numpy_time,
            "numba_seconds_per_epoch": numba_time,
            "numba_threads": threads[-1],
            "speedup": numpy_time / numba_time,
        }
        print(f"small data: NumPy {numpy_time:.2f} s/epoch, Numba "
              f"{numba_time:.3f} s/epoch")

    plot_lines(threads, {"measured": [speedup[t] for t in threads],
                         "ideal": [t / threads[0] for t in threads]},
               "threads", "speed-up", "Parallel speed-up of one ALS epoch",
               "speedup.png", args.output_dir / "plots")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "results.json").write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
