from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()


from recsys_data.RatingIndex import RatingRow


def _save_plot(fig: plt.Figure, filename: str, output_dir: str | Path | None = None) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_dir / filename)
    plt.close(fig)


def plot_rating_distribution(
    ratings: list[RatingRow],
    bins: int = 20,
    output_dir: str | Path | None = None,
):
    values = [row.rating for row in ratings]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(values, bins=np.linspace(0.5, 5.0, bins + 1))
    ax.set_title("Distribution of Movie Ratings")
    ax.set_xlabel("Rating")
    ax.set_ylabel("Count")
    ax.set_xlim(0.5, 5.0)
    _save_plot(fig, "rating_distribution.png", output_dir=output_dir)


def plot_loglog_degrees(
    user_degrees: np.ndarray,
    movie_degrees: np.ndarray,
    output_dir: str | Path | None = None,
):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, values, title in [
        (axes[0], user_degrees, "User degree distribution"),
        (axes[1], movie_degrees, "Movie degree distribution"),
    ]:
        unique_values, counts = np.unique(values, return_counts=True)
        # scatter plot
        log_unique_values = np.log10(unique_values)
        log_counts = np.log10(counts)
        ax.scatter(log_unique_values, log_counts)
        ax.set_title(title)
        ax.set_xlabel("Log10 Degree k")
        ax.set_ylabel("Log10 Count P(k)")

    _save_plot(fig, "degree_distribution.png", output_dir=output_dir)


def plot_two_metrics(
    iterations: list[int],
    train_metric: list[float],
    test_metric: list[float],
    title: str,
    ylabel: str,
    xlabel: str = "Training iteration",
    offset: int = 0,
    output_dir: str | Path | None = None,
):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(iterations[offset:], train_metric[offset:], label="Train")
    ax.plot(iterations[offset:], test_metric[offset:], label="Test")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()
    _save_plot(fig, f"{title.replace(' ', '_')}_metrics_{offset}.png", output_dir=output_dir)


def plot_objective(
    iterations: list[int],
    training_objective: list[float],
    offset: int = 0,
    title: str = "Negative Log Likelihood",
    output_dir: str | Path | None = None,
):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(iterations[offset:], training_objective[offset:])
    ax.set_title(title)
    ax.set_xlabel("Training iteration")
    ax.set_ylabel("Objective")
    ax.grid(True, alpha=0.3)
    _save_plot(fig, f"{title.replace(' ', '_')}_objective_{offset}.png", output_dir=output_dir)
