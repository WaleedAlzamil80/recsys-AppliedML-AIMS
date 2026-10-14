from pathlib import Path

import numpy as np

from recsys_data.data_loader import load_data
from recsys_data.data_spliter import DataSplitter, verification
from recsys_data.RatingIndex import RatingsIndex, RatingRow
from visulizations.plots import plot_objective, plot_two_metrics


def predict_rating(
    user_index: int,
    item_index: int,
    user_biases: np.ndarray,
    item_biases: np.ndarray,
    user_vectors: np.ndarray,
    item_vectors: np.ndarray,
) -> float:
    return (np.dot(user_vectors[user_index], item_vectors[item_index]) + user_biases[user_index] + item_biases[item_index])


def evaluate(
    data_by_user: dict[int, list[RatingRow]],
    user_biases: np.ndarray,
    item_biases: np.ndarray,
    user_vectors: np.ndarray,
    item_vectors: np.ndarray,
    item_id_to_index: dict[int, int],
    lamb: float,
    gamma: float,
    tau: float,
) -> tuple[float, float]:
    squared_error = 0.0
    count = 0
    user_indices = set()
    item_indices = set()

    for user_index, user_ratings in data_by_user.items():
        for row in user_ratings:
            item_index = item_id_to_index[row.movie_id]
            prediction = predict_rating(
                user_index,
                item_index,
                user_biases,
                item_biases,
                user_vectors,
                item_vectors,
            )
            squared_error += (row.rating - prediction) ** 2
            count += 1
            user_indices.add(user_index)
            item_indices.add(item_index)

    rmse = np.sqrt(squared_error / count) if count else np.nan
    nll = 0.5 * lamb * squared_error
    nll += 0.5 * gamma * (
        sum(user_biases[user_index] ** 2 for user_index in user_indices)
        + sum(item_biases[item_index] ** 2 for item_index in item_indices)
    )
    nll += 0.5 * tau * (
        sum(np.sum(user_vectors[user_index] ** 2) for user_index in user_indices)
        + sum(np.sum(item_vectors[item_index] ** 2) for item_index in item_indices)
    )
    return float(rmse), float(nll)


def fit_als(
    data_by_user_train: dict[int, list[RatingRow]],
    data_by_movie_train: dict[int, list[RatingRow]],
    data_by_user_test: dict[int, list[RatingRow]],
    user_id_to_index: dict[int, int],
    item_id_to_index: dict[int, int],
    epochs: int,
    latent_dim: int,
    lamb: float,
    gamma: float,
    tau: float,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[float], list[float], list[float], list[float]]:
    rng = np.random.default_rng(random_state)

    M = len(user_id_to_index)
    N = len(item_id_to_index)
    user_biases = np.zeros(M)
    item_biases = np.zeros(N)
    user_vectors = rng.normal(0.0, 0.1, (M, latent_dim))
    item_vectors = rng.normal(0.0, 0.1, (N, latent_dim))
    losses = []
    losses_test = []
    train_rmse = []
    test_rmse = []

    for _ in range(epochs):
        for user_index, user_ratings in data_by_user_train.items():
            if not user_ratings:
                continue

            # update the user bias
            bias_sum = lamb * sum(row.rating - np.dot(user_vectors[user_index], item_vectors[item_id_to_index[row.movie_id]]) - item_biases[item_id_to_index[row.movie_id]] for row in user_ratings)
            user_biases[user_index] = bias_sum / (lamb * len(user_ratings) + gamma)

            # update the user vector
            A = tau * np.eye(latent_dim)
            b = np.zeros(latent_dim)
            for row in user_ratings:
                item_index = item_id_to_index[row.movie_id]
                v_n = item_vectors[item_index]
                b_n = item_biases[item_index]
                b_m = user_biases[user_index]
                residual = row.rating - (b_m + b_n)
                A += lamb * np.outer(v_n, v_n)
                b += lamb * v_n * residual
            user_vectors[user_index] = np.linalg.solve(A, b)

        for item_index, movie_ratings in data_by_movie_train.items():
            if not movie_ratings:
                continue

            # update the item bias
            bias_sum = lamb * sum(row.rating - np.dot(user_vectors[user_id_to_index[row.user_id]], item_vectors[item_index]) - user_biases[user_id_to_index[row.user_id]] for row in movie_ratings)
            item_biases[item_index] = bias_sum / (lamb * len(movie_ratings) + gamma)

            # update the item vector
            A = tau * np.eye(latent_dim)
            b = np.zeros(latent_dim)
            for row in movie_ratings:
                user_index = user_id_to_index[row.user_id]
                u_m = user_vectors[user_index]
                b_m = user_biases[user_index]
                b_n = item_biases[item_index]
                residual = row.rating - (b_m + b_n)
                b += lamb * u_m * residual
                A += lamb * np.outer(u_m, u_m)
            item_vectors[item_index] = np.linalg.solve(A, b)

        # compute training RMSE and objective after each epoch
        current_train_rmse, loss = evaluate(
            data_by_user_train,
            user_biases,
            item_biases,
            user_vectors,
            item_vectors,
            item_id_to_index,
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
            item_id_to_index,
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

    user_index_to_id = sorted(train_index.get_all_users())
    item_index_to_id = sorted(train_index.get_all_movies())
    user_id_to_index = {user_id: index for index, user_id in enumerate(user_index_to_id)}
    item_id_to_index = {item_id: index for index, item_id in enumerate(item_index_to_id)}

    data_by_user_train = {index: train_index.get_user_ratings(user_id) for index, user_id in enumerate(user_index_to_id)}
    data_by_movie_train = {index: train_index.get_movie_ratings(movie_id) for index, movie_id in enumerate(item_index_to_id)}
    data_by_user_test = {index: test_index.get_user_ratings(user_id) for index, user_id in enumerate(user_index_to_id)}

    epochs = 30
    latent_dim = 10
    lamb = 0.09
    gamma = 0.1
    tau = 0.9
    output_dir = Path(__file__).resolve().parent / "output" / "practice_03"

    user_biases, item_biases, user_vectors, item_vectors, losses, losses_test, train_rmse, test_rmse = fit_als(
        data_by_user_train,
        data_by_movie_train,
        data_by_user_test,
        user_id_to_index,
        item_id_to_index,
        epochs,
        latent_dim,
        lamb,
        gamma,
        tau,
        random_state=42,
    )

    iterations = list(range(1, epochs + 1))
    plot_objective(iterations, losses, title="ALS Regularized Negative Log Likelihood", output_dir=output_dir / "plots", offset=1)
    plot_objective(iterations, losses_test, title="ALS Regularized Negative Log Likelihood (Test)", output_dir=output_dir / "plots", offset=1)
    plot_two_metrics(
        iterations,
        train_rmse,
        test_rmse,
        title="ALS Root Mean Squared Error",
        ylabel="RMSE",
        output_dir=output_dir / "plots",
        offset=1,
    )

    print(f"Final training objective: {losses[-1]:.4f}")
    print(f"Final training RMSE: {train_rmse[-1]:.4f}")
    print(f"Final test RMSE: {test_rmse[-1]:.4f}")
    print(f"Training objective non-increasing: {np.all(np.diff(losses) <= 1e-8)}")

    # save the npy arrays to the output directory
    (output_dir / "npy_arrays").mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "npy_arrays" / "user_biases.npy", user_biases)
    np.save(output_dir / "npy_arrays" / "item_biases.npy", item_biases)
    np.save(output_dir / "npy_arrays" / "user_vectors.npy", user_vectors)
    np.save(output_dir / "npy_arrays" / "item_vectors.npy", item_vectors)
    np.save(output_dir / "npy_arrays" / "user_ids.npy", np.asarray(user_index_to_id))
    np.save(output_dir / "npy_arrays" / "item_ids.npy", np.asarray(item_index_to_id))

if __name__ == "__main__":
    main()
