from pathlib import Path

import numpy as np

from recsys_data.data_loader import load_data
from recsys_data.data_spliter import DataSplitter, verification
from recsys_data.RatingIndex import RatingsIndex
from visulizations.plots import plot_objective, plot_two_metrics


def evaluate(
    data_by_user,
    user_bias: np.ndarray,
    item_bias: np.ndarray,
    user_id_to_index: dict[int, int],
    item_id_to_index: dict[int, int],
    lamb: float,
    gamma: float,
) -> tuple[float, float, float]:
    squared_error = 0.0
    count = 0
    user_indices = set()
    item_indices = set()
    for user_index, item_ratings in data_by_user.items():
        for row in item_ratings:
            item_index = item_id_to_index[row.movie_id]
            residual = row.rating - user_bias[user_index] - item_bias[item_index]
            squared_error += residual**2
            count += 1
            user_indices.add(user_index)
            item_indices.add(item_index)
    rmse = np.sqrt(squared_error / count) if count else np.nan
    nll = 0.5 * lamb * squared_error
    nll += 0.5 * gamma * (
        sum(user_bias[user_index] ** 2 for user_index in user_indices)
        + sum(item_bias[item_index] ** 2 for item_index in item_indices)
    )
    return float(rmse), squared_error, float(nll)


def main():
    DATA_DIR = Path(__file__).resolve().parents[0] / "datasets" / "ml-latest-small"
    ratings, movies, index = load_data(DATA_DIR)
    splitter = DataSplitter(ratings, test_ratio=0.2, random_state=42)
    train_data, test_data = splitter.stratified_split()
    train_index = RatingsIndex(train_data)
    test_index = RatingsIndex(test_data)

    # verify that all the users and movies must exist in the training set and check if they do in the test set
    verification(train_index, test_index)

    user_index_to_id = sorted(train_index.get_all_users())
    item_index_to_id = sorted(train_index.get_all_movies())
    print(f"Total users: {len(user_index_to_id)}, Total items: {len(item_index_to_id)}")
    print(f"First 5 users: {user_index_to_id[:5]}")
    print(f"First 5 items: {item_index_to_id[:5]}")

    user_id_to_index = {user_id: index for index, user_id in enumerate(user_index_to_id)}
    item_id_to_index = {item_id: index for index, item_id in enumerate(item_index_to_id)}
    M = len(user_index_to_id)
    N = len(item_index_to_id)
    print(f"First 5 user_id_to_index: {list(user_id_to_index.items())[:5]}")
    print(f"First 5 item_id_to_index: {list(item_id_to_index.items())[:5]}")

    print(f"Total users: {M}, Total items: {N}")

    data_by_user_train = {index: train_index.get_user_ratings(user_id) for index, user_id in enumerate(user_index_to_id)}
    data_by_user_test = {index: test_index.get_user_ratings(user_id) for index, user_id in enumerate(user_index_to_id)}
    data_by_item_train = {index: train_index.get_movie_ratings(item_id) for index, item_id in enumerate(item_index_to_id)}

    epochs = 100
    lamb = 0.9
    gamma = 0.1

    user_bias = np.zeros(M)
    item_bias = np.zeros(N)
    iterations = list(range(1, epochs + 1))
    negative_log_likelihood = []
    negative_log_likelihood_test = []
    train_rmse = []
    test_rmse = []
    output_dir = Path(__file__).resolve().parent / "output" / "practice_02"

    for epoch in range(epochs):
        # Update User Biases
        for user_index, item_ratings in data_by_user_train.items():
            if not item_ratings:
                continue
            bias_sum = sum(lamb * (row.rating - item_bias[item_id_to_index[row.movie_id]]) for row in item_ratings)
            user_bias[user_index] = bias_sum / (lamb * len(item_ratings) + gamma)

        # Update Item Biases
        for item_index, user_ratings in data_by_item_train.items():
            if not user_ratings:
                continue
            bias_sum = sum(lamb * (row.rating - user_bias[user_id_to_index[row.user_id]]) for row in user_ratings)
            item_bias[item_index] = bias_sum / (lamb * len(user_ratings) + gamma)

        # Evaluations and Metrics
        current_train_rmse, _, train_nll = evaluate(data_by_user_train, user_bias, item_bias, user_id_to_index, item_id_to_index, lamb, gamma)
        current_test_rmse, _, test_nll = evaluate(data_by_user_test, user_bias, item_bias, user_id_to_index, item_id_to_index, lamb, gamma)

        negative_log_likelihood.append(train_nll)
        negative_log_likelihood_test.append(test_nll)

        train_rmse.append(current_train_rmse)
        test_rmse.append(current_test_rmse)

    plot_two_metrics(iterations, train_rmse, test_rmse, title="ALS Root Mean Squared Error", ylabel="RMSE", output_dir=output_dir / "plots", offset=1)
    plot_objective(iterations, negative_log_likelihood, title="ALS Regularized Negative Log Likelihood", output_dir=output_dir / "plots", offset=1)
    plot_objective(iterations, negative_log_likelihood_test, title="ALS Regularized Negative Log Likelihood (Test)", output_dir=output_dir / "plots", offset=1)
    print(f"Final training RMSE: {train_rmse[-1]:.4f}")
    print(f"Final test RMSE: {test_rmse[-1]:.4f}")

    # save the npy arrays to the output directory
    (output_dir / "npy_arrays").mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "npy_arrays" / "user_bias.npy", user_bias)
    np.save(output_dir / "npy_arrays" / "item_bias.npy", item_bias)
    np.save(output_dir / "npy_arrays" / "user_ids.npy", np.asarray(user_index_to_id))
    np.save(output_dir / "npy_arrays" / "item_ids.npy", np.asarray(item_index_to_id))

if __name__ == "__main__":
    main()
