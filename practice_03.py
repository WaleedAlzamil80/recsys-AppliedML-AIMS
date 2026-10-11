from pathlib import Path

import numpy as np

from recsys_data.data_loader import load_data
from recsys_data.data_spliter import DataSplitter, verification
from recsys_data.RatingIndex import RatingsIndex, RatingRow
from visulizations.plots import plot_objective, plot_two_metrics


def predict_rating(
    user_id: int,
    movie_id: int,
    user_biases: np.ndarray,
    item_biases: np.ndarray,
    user_vectors: np.ndarray,
    item_vectors: np.ndarray,
) -> float:
    return np.dot(user_vectors[user_id - 1], item_vectors[movie_id - 1]) + user_biases[user_id - 1] + item_biases[movie_id - 1]


def evaluate(
    data_by_user: dict[int, list[RatingRow]],
    user_biases: np.ndarray,
    item_biases: np.ndarray,
    user_vectors: np.ndarray,
    item_vectors: np.ndarray,
    lamb: float,
    gamma: float,
    tau: float,
) -> tuple[float, float]:
    squared_error = 0.0
    count = 0
    user_ids = set()
    movie_ids = set()

    for user_id, user_ratings in data_by_user.items():
        for row in user_ratings:
            prediction = predict_rating(
                user_id,
                row.movie_id,
                user_biases,
                item_biases,
                user_vectors,
                item_vectors,
            )
            squared_error += (row.rating - prediction) ** 2
            count += 1
            user_ids.add(user_id)
            movie_ids.add(row.movie_id)

    rmse = np.sqrt(squared_error / count) if count else np.nan
    nll = 0.5 * lamb * squared_error
    nll += 0.5 * gamma * (
        sum(user_biases[user_id - 1] ** 2 for user_id in user_ids)
        + sum(item_biases[movie_id - 1] ** 2 for movie_id in movie_ids)
    )
    nll += 0.5 * tau * (
        sum(np.sum(user_vectors[user_id - 1] ** 2) for user_id in user_ids)
        + sum(np.sum(item_vectors[movie_id - 1] ** 2) for movie_id in movie_ids)
    )
    return float(rmse), float(nll)


def fit_als(
    data_by_user_train: dict[int, list[RatingRow]],
    data_by_movie_train: dict[int, list[RatingRow]],
    data_by_user_test: dict[int, list[RatingRow]],
    epochs: int,
    latent_dim: int,
    lamb: float,
    gamma: float,
    tau: float,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[float], list[float], list[float]]:
    rng = np.random.default_rng(random_state)

    M = max(data_by_user_train)
    N = max(data_by_movie_train)
    user_biases = np.zeros(M)
    item_biases = np.zeros(N)
    user_vectors = rng.normal(0.0, 0.1, (M, latent_dim))
    item_vectors = rng.normal(0.0, 0.1, (N, latent_dim))
    losses = []
    losses_test = []
    train_rmse = []
    test_rmse = []

    for _ in range(epochs):
        for user_id in range(1, M + 1):
            user_ratings = data_by_user_train.get(user_id, [])
            if not user_ratings:
                continue

            # update the user bias
            bias_sum = lamb * sum((row.rating - np.dot(user_vectors[user_id - 1], item_vectors[row.movie_id - 1]) - item_biases[row.movie_id - 1]) for row in user_ratings)
            user_biases[user_id - 1] = bias_sum / (lamb * len(user_ratings) + gamma)

            # update the user vector
            A = tau * np.eye(latent_dim)
            b = np.zeros(latent_dim)
            for row in user_ratings:
                item_id = row.movie_id
                v_n = item_vectors[item_id - 1]
                b_n = item_biases[item_id - 1]
                b_m = user_biases[user_id - 1]
                residual = row.rating - (b_m + b_n)
                A += lamb * np.outer(v_n, v_n)
                b += lamb * v_n * residual
            # user_vectors[user_id - 1] = np.dot(np.linalg.pinv(A), b)
            user_vectors[user_id - 1] = np.linalg.solve(A, b)

        for movie_id in range(1, N + 1):
            movie_ratings = data_by_movie_train.get(movie_id, [])
            if not movie_ratings:
                continue

            # update the item bias
            bias_sum = lamb * sum((row.rating - np.dot(user_vectors[row.user_id - 1], item_vectors[movie_id - 1]) - user_biases[row.user_id - 1]) for row in movie_ratings)
            item_biases[movie_id - 1] = bias_sum / (lamb * len(movie_ratings) + gamma)

            # update the item vector
            A = tau * np.eye(latent_dim)
            b = np.zeros(latent_dim)
            for row in movie_ratings:
                user_id = row.user_id
                u_m = user_vectors[user_id - 1]
                b_m = user_biases[user_id - 1]
                b_n = item_biases[movie_id - 1]
                residual = row.rating - (b_m + b_n)
                b += lamb * u_m * residual
                A += lamb * np.outer(u_m, u_m)
            # item_vectors[movie_id - 1] = np.dot(np.linalg.pinv(A), b)
            item_vectors[movie_id - 1] = np.linalg.solve(A, b)

        # compute training RMSE and objective after each epoch
        current_train_rmse, loss = evaluate(
            data_by_user_train,
            user_biases,
            item_biases,
            user_vectors,
            item_vectors,
            lamb,
            gamma,
            tau,
        )
        losses.append(loss)
        train_rmse.append(current_train_rmse)
        current_test_rmse, loss_test = evaluate(
            data_by_user_test,
            user_biases,
            item_biases,
            user_vectors,
            item_vectors,
            lamb,
            gamma,
            tau,
        )
        losses_test.append(loss_test)
        test_rmse.append(current_test_rmse)

    return user_biases, item_biases, user_vectors, item_vectors, losses, losses_test, train_rmse, test_rmse


def main() -> None:
    data_dir = Path(__file__).resolve().parent / "datasets" / "ml-latest-small"
    ratings, _, _ = load_data(data_dir)
    train_data, test_data = DataSplitter(ratings, test_ratio=0.2, random_state=42).stratified_split()
    train_index = RatingsIndex(train_data)
    test_index = RatingsIndex(test_data)
    verification(train_index, test_index)

    M = max(row.user_id for row in ratings) + 1
    N = max(row.movie_id for row in ratings) + 1

    data_by_user_train = {user_id: train_index.get_user_ratings(user_id) for user_id in range(1, M + 1)}
    data_by_movie_train = {movie_id: train_index.get_movie_ratings(movie_id) for movie_id in range(1, N + 1)}
    data_by_user_test = {user_id: test_index.get_user_ratings(user_id) for user_id in range(1, M + 1)}

    epochs = 30
    latent_dim = 10
    lamb = 0.9
    gamma = 0.1
    tau = 0.1
    output_dir = Path(__file__).resolve().parent / "output" / "practice_03"

    user_biases, item_biases, user_vectors, item_vectors, losses, losses_test, train_rmse, test_rmse = fit_als(
        data_by_user_train,
        data_by_movie_train,
        data_by_user_test,
        epochs,
        latent_dim,
        lamb,
        gamma,
        tau,
        random_state=42,
    )

    iterations = list(range(1, epochs + 1))
    plot_objective(iterations, losses, title="ALS Regularized Negative Log Likelihood", output_dir=output_dir, offset=0)
    plot_objective(iterations, losses_test, title="ALS Regularized Negative Log Likelihood (Test)", output_dir=output_dir, offset=0)
    plot_two_metrics(
        iterations,
        train_rmse,
        test_rmse,
        title="ALS Root Mean Squared Error",
        ylabel="RMSE",
        output_dir=output_dir,
        offset=0,
    )

    print(f"Final training objective: {losses[-1]:.4f}")
    print(f"Final training RMSE: {train_rmse[-1]:.4f}")
    print(f"Final test RMSE: {test_rmse[-1]:.4f}")
    print(f"Training objective non-increasing: {np.all(np.diff(losses) <= 1e-8)}")


if __name__ == "__main__":
    main()
