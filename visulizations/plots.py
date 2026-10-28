from pathlib import Path
from typing import Sequence

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()


from recsys_data.RatingIndex import RatingRow


def _save_plot(fig: plt.Figure, filename: str, output_dir: str | Path | None = None) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_dir / filename, dpi=200)
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
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.2))

    for ax, values, title in [
        (axes[0], user_degrees, "User degree distribution"),
        (axes[1], movie_degrees, "Movie degree distribution"),
    ]:
        # logarithmic bins: raw counts are too sparse in the tail to read
        bins = np.unique(np.round(np.logspace(
            np.log10(values.min()), np.log10(values.max() + 1), 25)))
        density, edges = np.histogram(values, bins=bins, density=True)
        centres = np.sqrt(edges[:-1] * edges[1:])
        keep = density > 0
        ax.loglog(centres[keep], density[keep], "o", ms=4)
        ax.set_title(title)
        ax.set_xlabel("Degree k")
        ax.set_ylabel("P(k) (log-binned)")

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


# ---------------------------------------------------------------------------
# Figures for the report (practicals 5-10). Sizes are chosen so that a figure
# fits one column of the two-column ICML layout.
# ---------------------------------------------------------------------------
COLUMN_SIZE = (4.6, 3.3)


def plot_lines(
    x: Sequence[float],
    series: dict[str, Sequence[float]],
    xlabel: str,
    ylabel: str,
    title: str,
    filename: str,
    output_dir: str | Path,
    logx: bool = False,
    logy: bool = False,
    markers: bool = True,
    figsize: tuple[float, float] = COLUMN_SIZE,
) -> None:
    """Plot several named curves against a shared x axis."""
    fig, ax = plt.subplots(figsize=figsize)
    styles = ["-", "--", ":"]
    for i, (name, values) in enumerate(series.items()):
        ax.plot(x, values, marker="o" if markers else None, ms=3.5,
                ls=styles[i % len(styles)], lw=2.2 - 0.5 * i, label=name)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if logx:
        ax.set_xscale("log")
        if len(x) <= 10:
            ax.set_xticks(list(x))
            ax.set_xticklabels([f"{v:g}" for v in x])
            ax.minorticks_off()
    if logy:
        ax.set_yscale("log")
    if len(series) > 1:
        ax.legend(fontsize=8)
    _save_plot(fig, filename, output_dir=Path(output_dir))


def plot_heatmap(
    matrix: np.ndarray,
    row_labels: Sequence[str],
    col_labels: Sequence[str],
    xlabel: str,
    ylabel: str,
    title: str,
    filename: str,
    output_dir: str | Path,
    fmt: str = ".3f",
    figsize: tuple[float, float] = COLUMN_SIZE,
) -> None:
    """Annotated heat map (e.g. validation RMSE over a hyper-parameter grid)."""
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(matrix, annot=True, fmt=fmt, cmap="viridis_r",
                xticklabels=list(col_labels), yticklabels=list(row_labels),
                annot_kws={"fontsize": 7}, ax=ax,
                cbar_kws={"label": "validation RMSE"})
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    _save_plot(fig, filename, output_dir=Path(output_dir))


def plot_embedding_2d(
    coords: np.ndarray,
    groups: Sequence[int],
    group_names: Sequence[str],
    annotations: Sequence[tuple[int, str]],
    title: str,
    filename: str,
    output_dir: str | Path,
    figsize: tuple[float, float] = (4.8, 4.6),
) -> None:
    """Scatter plot of 2D trait vectors coloured by a group (e.g. genre).

    ``annotations`` is a list of ``(row index, label)`` pairs to name.
    """
    fig, ax = plt.subplots(figsize=figsize)
    palette = sns.color_palette("tab10", n_colors=max(len(group_names), 1))
    groups = np.asarray(groups)
    for g, name in enumerate(group_names):
        chosen = groups == g
        if chosen.any():
            ax.scatter(coords[chosen, 0], coords[chosen, 1], s=9, alpha=0.6,
                       color=palette[g], label=name)
    other = groups < 0
    if other.any():
        ax.scatter(coords[other, 0], coords[other, 1], s=6, alpha=0.25,
                   color="gray", label="other")
    for row, label in list(annotations)[:12]:
        ax.annotate(label, coords[row], fontsize=6.5, alpha=0.9,
                    xytext=(3, 3), textcoords="offset points")
    ax.axhline(0, color="k", lw=0.4)
    ax.axvline(0, color="k", lw=0.4)
    ax.set_xlabel("trait 1")
    ax.set_ylabel("trait 2")
    ax.set_title(title)
    ax.legend(fontsize=6.5, markerscale=1.5, loc="upper center",
              bbox_to_anchor=(0.5, -0.14), ncol=4, frameon=False)
    _save_plot(fig, filename, output_dir=Path(output_dir))


def plot_genre_vectors(
    vectors: np.ndarray,
    names: Sequence[str],
    title: str,
    filename: str,
    output_dir: str | Path,
    figsize: tuple[float, float] = (4.8, 4.2),
) -> None:
    """Draw 2D genre vectors ``f_t`` as labelled arrows from the origin."""
    fig, ax = plt.subplots(figsize=figsize)
    palette = sns.color_palette("husl", n_colors=len(names))
    for vec, name, color in zip(vectors, names, palette):
        ax.annotate("", xy=vec, xytext=(0, 0),
                    arrowprops={"arrowstyle": "->", "color": color, "lw": 1})
        ax.annotate(name, vec, fontsize=7, color=color, xytext=(2, 2),
                    textcoords="offset points")
    limit = 1.25 * np.abs(vectors).max()
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
    ax.axhline(0, color="k", lw=0.4)
    ax.axvline(0, color="k", lw=0.4)
    ax.set_xlabel("trait 1")
    ax.set_ylabel("trait 2")
    ax.set_title(title)
    _save_plot(fig, filename, output_dir=Path(output_dir))


def plot_ccdf(
    degrees: dict[str, np.ndarray],
    fits: dict[str, tuple[float, float]],
    filename: str,
    output_dir: str | Path,
    figsize: tuple[float, float] = COLUMN_SIZE,
) -> None:
    """Empirical complementary CDFs on log-log axes with power-law fits.

    ``fits[name] = (alpha, x_min)``; the fitted tail ``(x / x_min)^(1-alpha)``
    is drawn from ``x_min`` onwards, anchored at the empirical CCDF.
    """
    fig, ax = plt.subplots(figsize=figsize)
    for name, values in degrees.items():
        xs = np.sort(np.asarray(values, dtype=float))
        ccdf = 1.0 - np.arange(len(xs)) / len(xs)
        line, = ax.loglog(xs, ccdf, ".", ms=3, alpha=0.7, label=name)
        alpha, x_min = fits[name]
        if np.isfinite(alpha):
            tail = xs[xs >= x_min]
            anchor = np.mean(xs >= x_min)
            ax.loglog(tail, anchor * (tail / x_min) ** (1.0 - alpha), "-",
                      color=line.get_color(), lw=1.2,
                      label=f"{name} fit, alpha={alpha:.2f}")
    ax.set_xlabel("degree k (number of ratings)")
    ax.set_ylabel("P(degree >= k)")
    ax.legend(fontsize=7)
    _save_plot(fig, filename, output_dir=Path(output_dir))


def plot_histogram(
    values: Sequence[float],
    xlabel: str,
    filename: str,
    output_dir: str | Path,
    bins: int = 20,
    vline: float | None = None,
    vline_label: str = "",
    figsize: tuple[float, float] = COLUMN_SIZE,
    log10: bool = False,
) -> None:
    """Histogram with an optional labelled vertical line.

    With ``log10=True`` the values (e.g. p-values that are all tiny) and the
    vertical line are shown on a log10 scale.
    """
    fig, ax = plt.subplots(figsize=figsize)
    if log10:
        values = np.log10(np.clip(np.asarray(values, dtype=float), 1e-300, None))
        vline = None if vline is None else float(np.log10(vline))
    ax.hist(values, bins=bins, edgecolor="white")
    if vline is not None:
        ax.axvline(vline, color="crimson", lw=1.5, label=vline_label)
        ax.legend(fontsize=8)
    ax.set_xlabel(f"log10 of {xlabel}" if log10 else xlabel)
    ax.set_ylabel("count")
    _save_plot(fig, filename, output_dir=Path(output_dir))



def plot_bars(
    labels: Sequence[str],
    values: Sequence[float],
    errors: Sequence[float],
    ylabel: str,
    filename: str,
    output_dir: str | Path,
    figsize: tuple[float, float] = (3.2, 3.0),
) -> None:
    """Bar chart with symmetric error bars (e.g. CTR with 95% intervals)."""
    fig, ax = plt.subplots(figsize=figsize)
    colors = ["#4c72b0", "#dd8452", "#55a868"][:len(values)]
    ax.bar(labels, values, yerr=errors, capsize=5, color=colors)
    ax.set_ylabel(ylabel)
    _save_plot(fig, filename, output_dir=Path(output_dir))


def plot_scatter(
    x: np.ndarray,
    y: np.ndarray,
    xlabel: str,
    ylabel: str,
    filename: str,
    output_dir: str | Path,
    logx: bool = False,
    alpha: float = 0.35,
    figsize: tuple[float, float] = COLUMN_SIZE,
) -> None:
    """Plain scatter plot."""
    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(x, y, s=6, alpha=alpha)
    if logx:
        ax.set_xscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    _save_plot(fig, filename, output_dir=Path(output_dir))
