from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from scipy.stats import linregress

from recsys_data.RatingIndex import RatingsIndex
from recsys_data.data_loader import load_data
from visulizations.plots import plot_loglog_degrees, plot_rating_distribution


def summarize_degree_distribution(index: RatingsIndex) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    user_degrees = np.array([len(v) for v in index.by_user.values()], dtype=float)
    movie_degrees = np.array([len(v) for v in index.by_movie.values()], dtype=float)
    return user_degrees, movie_degrees, np.concatenate([user_degrees, movie_degrees])


def estimate_power_law_slope(values: np.ndarray) -> float:
    unique_values, counts = np.unique(values, return_counts=True)

    xs = np.log10(np.asarray(unique_values, dtype=float))
    ys = np.log10(np.asarray(counts, dtype=float))

    cut = np.percentile(np.asarray(unique_values, dtype=float), 80)
    mask = np.asarray(unique_values, dtype=float) >= cut
    if mask.sum() < 2:
        return np.nan

    slope, _, _, _, _ = linregress(xs[mask], ys[mask])
    return slope


def main() -> None:
    DATA_DIR = Path(__file__).resolve().parents[0] / "datasets" / "ml-latest-small"
    ratings, movies, index = load_data(DATA_DIR)

    user_counts = [len(user_ratings) for user_ratings in index.users().values()]
    movie_counts = [len(movie_ratings) for movie_ratings in index.movies().values()]

    print(f"Loaded {len(ratings)} ratings across {len(user_counts)} users and {len(movie_counts)} movies.")
    print(f"User minimum ratings: {min(user_counts)}; maximum ratings: {max(user_counts)}")
    print(f"Movie minimum ratings: {min(movie_counts)}; maximum ratings: {max(movie_counts)}")
    output_dir = Path(__file__).resolve().parent / "output" / "practice_01"
    plot_rating_distribution(ratings, output_dir=output_dir)

    user_degrees, movie_degrees, combined = summarize_degree_distribution(index)
    plot_loglog_degrees(user_degrees, movie_degrees, output_dir=output_dir)

    # with ProcessPoolExecutor(max_workers=2) as workers:
    #     user_job = workers.submit(estimate_power_law_slope, user_degrees)
    #     movie_job = workers.submit(estimate_power_law_slope, movie_degrees)
    #     user_slope = user_job.result()
    #     movie_slope = movie_job.result()
    user_slope = estimate_power_law_slope(user_degrees)
    movie_slope = estimate_power_law_slope(movie_degrees)

    print(f"Estimated log-log slope for user degree tail: {user_slope:.3f}")
    print(f"Estimated log-log slope for movie degree tail: {movie_slope:.3f}")

    print("\nInterpretation:")
    print("- MovieLens ratings are highly skewed, but the rating values themselves are bounded between 0.5 and 5.0, so they are not a power-law quantity.")
    print("- The degree distribution of the user-movie bipartite graph is truncated by design: each user rated at least 20 movies, so the lower tail is not representative of a natural scale-free process.")
    print("- The movie degree distribution is also bounded by the finite number of users and by repeated interactions, which prevents a true scale-free tail from extending indefinitely.")
    print("- In short: the data shows heavy-tailed popularity patterns, but not a clean scale-free network in the strict sense because of the sampling/truncation constraints.")


if __name__ == "__main__":
    main()
