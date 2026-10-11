from collections import defaultdict
from typing import NamedTuple


class RatingRow(NamedTuple):
    user_id: int
    movie_id: int
    rating: float
    timestamp: int


class MovieRow(NamedTuple):
    movie_id: int
    title: str
    genres: str


class RatingsIndex:

    def __init__(self, ratings: list[RatingRow]):
        self.by_user: defaultdict[int, list[RatingRow]] = defaultdict(list)
        self.by_movie: defaultdict[int, list[RatingRow]] = defaultdict(list)

        for row in ratings:
            self.by_user[row.user_id].append(row)
            self.by_movie[row.movie_id].append(row)

    def users(self):
        return self.by_user
    
    def get_all_users(self) -> set[int]:
        return set(self.by_user.keys())
    
    def get_all_movies(self) -> set[int]:
        return set(self.by_movie.keys())

    def get_user_ratings(self, user_id: int) -> list[RatingRow]:
        return self.by_user.get(user_id, [])

    def movies(self):
        return self.by_movie

    def get_movie_ratings(self, movie_id: int) -> list[RatingRow]:
        return self.by_movie.get(movie_id, [])

