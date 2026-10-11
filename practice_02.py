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
    lamb: float,
    gamma: float,
) -> tuple[float, float, float]:
    squared_error = 0.0
    count = 0
    user_ids = set()
    item_ids = set()
    for user_id, item_ratings in data_by_user.items():
        for row in item_ratings:
            residual = row.rating - user_bias[user_id - 1] - item_bias[row.movie_id - 1]
            squared_error += residual**2
            count += 1
            user_ids.add(user_id)
            item_ids.add(row.movie_id)
    rmse = np.sqrt(squared_error / count) if count else np.nan
    nll = 0.5 * lamb * squared_error
    nll += 0.5 * gamma * (
        sum(user_bias[user_id - 1] ** 2 for user_id in user_ids)
        + sum(item_bias[item_id - 1] ** 2 for item_id in item_ids)
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

    M = max(row.user_id for row in ratings) + 1   # num of users
    N = max(row.movie_id for row in ratings) + 1   # num of items

    print(f"Total users: {M}, Total items: {N}")

    data_by_user_train = {user_id: train_index.get_user_ratings(user_id) for user_id in range(1, M + 1)}
    data_by_user_test = {user_id: test_index.get_user_ratings(user_id) for user_id in range(1, M + 1)}
    data_by_item_train = {item_id: train_index.get_movie_ratings(item_id) for item_id in range(1, N + 1)}
    data_by_item_test = {item_id: test_index.get_movie_ratings(item_id) for item_id in range(1, N + 1)}

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
        for u in range(1, M + 1):
            item_ratings = data_by_user_train[u]
            if not item_ratings:
                continue
            bias_sum = sum(lamb * (row.rating - item_bias[row.movie_id - 1]) for row in item_ratings)
            user_bias[u - 1] = bias_sum / (lamb * len(item_ratings) + gamma)

        # Update Item Biases
        for m in range(1, N + 1):
            user_ratings = data_by_item_train[m]
            if not user_ratings:
                continue
            bias_sum = sum(lamb * (row.rating - user_bias[row.user_id - 1]) for row in user_ratings)            
            item_bias[m - 1] = bias_sum / (lamb * len(user_ratings) + gamma)

        # Evaluations and Metrics
        current_train_rmse, _, train_nll = evaluate(data_by_user_train, user_bias, item_bias, lamb, gamma)
        current_test_rmse, _, test_nll = evaluate(data_by_user_test, user_bias, item_bias, lamb, gamma)

        negative_log_likelihood.append(train_nll)
        negative_log_likelihood_test.append(test_nll)

        train_rmse.append(current_train_rmse)
        test_rmse.append(current_test_rmse)

    plot_two_metrics(iterations, train_rmse, test_rmse, title="ALS Root Mean Squared Error", ylabel="RMSE", output_dir=output_dir, offset=0)
    plot_objective(iterations, negative_log_likelihood, title="ALS Regularized Negative Log Likelihood", output_dir=output_dir, offset=0)
    plot_objective(iterations, negative_log_likelihood_test, title="ALS Regularized Negative Log Likelihood (Test)", output_dir=output_dir, offset=0)
    print(f"Final training RMSE: {train_rmse[-1]:.4f}")
    print(f"Final test RMSE: {test_rmse[-1]:.4f}")

if __name__ == "__main__":
    main()
