"""Generate a small synthetic MovieLens-like dataset for smoke tests.

The data is NOT real. It only mimics the file layout (``ratings.csv`` and
``movies.csv``), the power-law item popularity, the 0.5-step rating scale and
genre-correlated latent structure of MovieLens, so every script in this
repository can be tested without downloading the real data.

Example:
    python tools/make_synthetic_movielens.py --out datasets/ml-synthetic
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

GENRES = [
    "Action", "Adventure", "Animation", "Children", "Comedy", "Crime",
    "Documentary", "Drama", "Fantasy", "Horror", "Musical", "Mystery",
    "Romance", "Sci-Fi", "Thriller", "War", "Western",
]
LOTR = [
    "The Lord of the Rings: The Fellowship of the Ring (2001)",
    "The Lord of the Rings: The Two Towers (2002)",
    "The Lord of the Rings: The Return of the King (2003)",
]


def build_movies(n_items: int, rng: np.random.Generator) -> pd.DataFrame:
    titles = [f"Synthetic Movie {i} ({1950 + i % 70})" for i in range(n_items)]
    genres = []
    for _ in range(n_items):
        count = rng.choice([0, 1, 2, 3], p=[0.02, 0.38, 0.40, 0.20])
        chosen = rng.choice(GENRES, size=count, replace=False)
        genres.append("|".join(chosen) if count else "(no genres listed)")
    # The three LOTR films are the most popular items: ids 0, 1 and 2.
    for position, title in enumerate(LOTR):
        titles[position] = title
        genres[position] = "Adventure|Fantasy"
    return pd.DataFrame(
        {"movieId": np.arange(1, n_items + 1), "title": titles,
         "genres": genres}
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--users", type=int, default=600)
    parser.add_argument("--items", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    movies = build_movies(args.items, rng)

    latent_dim = 4
    genre_vecs = {g: rng.normal(0, 0.8, latent_dim) for g in GENRES}
    item_vecs = np.zeros((args.items, latent_dim))
    for n, genre_string in enumerate(movies["genres"]):
        parts = [g for g in genre_string.split("|") if g in genre_vecs]
        base = np.mean([genre_vecs[g] for g in parts], axis=0) if parts else 0
        item_vecs[n] = base + rng.normal(0, 0.35, latent_dim)
    user_vecs = rng.normal(0, 0.7, (args.users, latent_dim))
    item_bias = rng.normal(0, 0.5, args.items)
    user_bias = rng.normal(0, 0.4, args.users)

    popularity = 1.0 / np.arange(1, args.items + 1) ** 0.9
    popularity /= popularity.sum()
    rows = []
    for m in range(args.users):
        count = int(min(args.items // 2, 20 + rng.lognormal(4.6, 0.9)))
        items = rng.choice(args.items, size=count, replace=False,
                           p=popularity)
        # Users tend to rate what they like: bias the choice by affinity.
        raw = (user_vecs[m] @ item_vecs[items].T) + item_bias[items]
        keep = rng.random(count) < 1 / (1 + np.exp(-(raw + 0.8)))
        keep[:20] = True
        items = items[keep]
        signal = (3.5 + user_bias[m] + item_bias[items]
                  + user_vecs[m] @ item_vecs[items].T
                  + rng.normal(0, 0.6, len(items)))
        values = np.clip(np.round(signal * 2) / 2, 0.5, 5.0)
        stamp = rng.integers(9e8, 1.6e9, len(items))
        rows.append(pd.DataFrame({"userId": m + 1, "movieId": items + 1,
                                  "rating": values, "timestamp": stamp}))
    ratings = pd.concat(rows, ignore_index=True)

    args.out.mkdir(parents=True, exist_ok=True)
    ratings.to_csv(args.out / "ratings.csv", index=False)
    movies.to_csv(args.out / "movies.csv", index=False)
    print(f"Wrote {len(ratings):,} synthetic ratings, {args.users} users, "
          f"{ratings['movieId'].nunique()} rated movies to {args.out}")


if __name__ == "__main__":
    main()
