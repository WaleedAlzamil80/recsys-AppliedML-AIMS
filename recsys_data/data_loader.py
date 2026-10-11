# import sys
# sys.path.append(".")  # Add the parent directory to the Python path

import csv
from pathlib import Path

from .RatingIndex import MovieRow, RatingRow, RatingsIndex
from .data_spliter import DataSplitter

def load_data(DATA_DIR: Path) -> tuple[list[RatingRow], list[MovieRow], RatingsIndex]:
    RATINGS_PATH = DATA_DIR / "ratings.csv"
    MOVIES_PATH = DATA_DIR / "movies.csv"

    ratings: list[RatingRow] = []
    reader = csv.reader(RATINGS_PATH.open(newline="", encoding="utf-8"))
    next(reader, None)
    for row in reader:
        if not row:
            continue
        ratings.append(
            RatingRow(
                user_id=int(row[0]),
                movie_id=int(row[1]),
                rating=float(row[2]),
                timestamp=int(row[3]),
            )
        )

    movies: list[MovieRow] = []
    reader = csv.reader(MOVIES_PATH.open(newline="", encoding="utf-8"))
    next(reader, None)
    for row in reader:
        if not row:
            continue
        title = row[1]
        genres = row[2] if len(row) > 2 else ""
        movies.append(MovieRow(movie_id=int(row[0]), title=title, genres=genres))

    return ratings, movies, RatingsIndex(ratings)


if __name__ == "__main__":
    DATA_DIR = Path(__file__).resolve().parents[1] / "datasets" / "ml-latest-small"
    print(f"Loading data... {DATA_DIR}")
    load_ratings, load_movies, load_ratings_index = load_data(DATA_DIR)
    print(f"Loaded {len(load_ratings)} ratings and {len(load_movies)} movies.")
    print(f"Loaded {len(load_ratings_index.users())} unique users.")
    print(f"Loaded {len(load_ratings_index.movies())} unique movies.")
    print(f"First rating: {load_ratings[0]}")
    print(f"First movie: {load_movies[0]}")
    # print(f"First user ratings: {load_ratings_index.get_user_ratings(load_ratings[0].user_id)}")
    # print(f"First movie ratings: {load_ratings_index.get_movie_ratings(load_ratings[0].movie_id)}")
    DataSplitterInstance = DataSplitter(load_ratings, test_ratio=0.2)
    train_ratings, test_ratings = DataSplitterInstance.split()
    print(f"Train ratings: {len(train_ratings)}")
    print(f"Test ratings: {len(test_ratings)}")