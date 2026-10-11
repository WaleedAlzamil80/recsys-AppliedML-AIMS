import numpy as np
from .RatingIndex import RatingRow, RatingsIndex
from collections import Counter

class DataSplitter:
    def __init__(self, ratings: list[RatingRow], test_ratio: float = 0.2, random_state: int = 42):
        self.ratings = ratings
        self.test_ratio = test_ratio
        self.random_state = random_state

    def split(self) -> tuple[list[RatingRow], list[RatingRow]]:
        # Shuffle the ratings to ensure randomness
        shuffled_ratings = self.ratings.copy()
        np.random.seed(self.random_state)
        np.random.shuffle(shuffled_ratings)

        # Calculate the split index
        split_index = int(len(shuffled_ratings) * (1 - self.test_ratio))

        # Split the ratings into training and test sets
        train_ratings = shuffled_ratings[:split_index]
        test_ratings = shuffled_ratings[split_index:]

        return train_ratings, test_ratings

    def stratified_split(self) -> tuple[list[RatingRow], list[RatingRow]]:
        if not 0 <= self.test_ratio < 1:
            raise ValueError("test_ratio must be in the range [0, 1)")

        rng = np.random.default_rng(self.random_state)
        shuffled_ratings = [self.ratings[i] for i in rng.permutation(len(self.ratings))]
        remaining_by_user = Counter(row.user_id for row in shuffled_ratings)
        remaining_by_movie = Counter(row.movie_id for row in shuffled_ratings)
        target_test_size = int(len(shuffled_ratings) * self.test_ratio)
        train_ratings = []
        test_ratings = []

        for row in shuffled_ratings:
            if (
                len(test_ratings) < target_test_size
                and remaining_by_user[row.user_id] > 1
                and remaining_by_movie[row.movie_id] > 1
            ):
                test_ratings.append(row)
                remaining_by_user[row.user_id] -= 1
                remaining_by_movie[row.movie_id] -= 1
            else:
                train_ratings.append(row)

        return train_ratings, test_ratings


def verification(train_index: RatingsIndex, test_index: RatingsIndex):
    train_user_ids = train_index.get_all_users()
    train_movie_ids = train_index.get_all_movies()
    test_user_ids = test_index.get_all_users()
    test_movie_ids = test_index.get_all_movies()

    missing_users = test_user_ids - train_user_ids
    missing_movies = test_movie_ids - train_movie_ids

    missing_users_train = train_user_ids - test_user_ids
    missing_movies_train = train_movie_ids - test_movie_ids

    # print numbers and rations
    if missing_users:
        print(f"Warning: {len(missing_users)} users are in the test set but not in the training set.")
    if missing_movies:
        print(f"Warning: {len(missing_movies)} movies are in the test set but not in the training set.")
    if missing_users_train:
        print(f"Note: {len(missing_users_train)} users are in the training set but not in the test set.")
    if missing_movies_train:
        print(f"Note: {len(missing_movies_train)} movies are in the training set but not in the test set.")

