from dataclasses import dataclass

import numba
import numpy as np
from numba import njit, prange

from .data import SparseRatings


@dataclass
class ALSResult:
    user_biases: np.ndarray
    item_biases: np.ndarray
    user_vectors: np.ndarray
    item_vectors: np.ndarray
    losses: list[float]
    losses_test: list[float]
    train_rmse: list[float]
    test_rmse: list[float]


@njit(parallel=True, cache=True)
def _update_users(
    indptr: np.ndarray,
    item_indices: np.ndarray,
    ratings: np.ndarray,
    user_biases: np.ndarray,
    item_biases: np.ndarray,
    user_vectors: np.ndarray,
    item_vectors: np.ndarray,
    lamb: float,
    gamma: float,
    tau: float,
) -> None:
    latent_dim = user_vectors.shape[1]
    for user_index in prange(user_biases.size):
        start = indptr[user_index]
        stop = indptr[user_index + 1]
        rating_count = stop - start
        if rating_count == 0:
            continue

        bias_sum = 0.0
        for rating_index in range(start, stop):
            item_index = item_indices[rating_index]
            prediction = item_biases[item_index]
            for factor_index in range(latent_dim):
                prediction += user_vectors[user_index, factor_index] * item_vectors[item_index, factor_index]
            bias_sum += lamb * (ratings[rating_index] - prediction)
        user_bias = bias_sum / (lamb * rating_count + gamma)
        user_biases[user_index] = user_bias

        system = np.zeros((latent_dim, latent_dim), dtype=np.float64)
        target = np.zeros(latent_dim, dtype=np.float64)
        for factor_index in range(latent_dim):
            system[factor_index, factor_index] = tau

        for rating_index in range(start, stop):
            item_index = item_indices[rating_index]
            residual = ratings[rating_index] - user_bias - item_biases[item_index]
            for left_factor in range(latent_dim):
                item_value_left = item_vectors[item_index, left_factor]
                target[left_factor] += lamb * item_value_left * residual
                for right_factor in range(latent_dim):
                    system[left_factor, right_factor] += (
                        lamb * item_value_left * item_vectors[item_index, right_factor]
                    )

        user_vectors[user_index, :] = np.linalg.solve(system, target)


@njit(parallel=True, cache=True)
def _update_items(
    indptr: np.ndarray,
    user_indices: np.ndarray,
    ratings: np.ndarray,
    user_biases: np.ndarray,
    item_biases: np.ndarray,
    user_vectors: np.ndarray,
    item_vectors: np.ndarray,
    lamb: float,
    gamma: float,
    tau: float,
) -> None:
    latent_dim = item_vectors.shape[1]
    for item_index in prange(item_biases.size):
        start = indptr[item_index]
        stop = indptr[item_index + 1]
        rating_count = stop - start
        if rating_count == 0:
            continue

        bias_sum = 0.0
        for rating_index in range(start, stop):
            user_index = user_indices[rating_index]
            prediction = user_biases[user_index]
            for factor_index in range(latent_dim):
                prediction += user_vectors[user_index, factor_index] * item_vectors[item_index, factor_index]
            bias_sum += lamb * (ratings[rating_index] - prediction)
        item_bias = bias_sum / (lamb * rating_count + gamma)
        item_biases[item_index] = item_bias

        system = np.zeros((latent_dim, latent_dim), dtype=np.float64)
        target = np.zeros(latent_dim, dtype=np.float64)
        for factor_index in range(latent_dim):
            system[factor_index, factor_index] = tau

        for rating_index in range(start, stop):
            user_index = user_indices[rating_index]
            residual = ratings[rating_index] - user_biases[user_index] - item_bias
            for left_factor in range(latent_dim):
                user_value_left = user_vectors[user_index, left_factor]
                target[left_factor] += lamb * user_value_left * residual
                for right_factor in range(latent_dim):
                    system[left_factor, right_factor] += (
                        lamb * user_value_left * user_vectors[user_index, right_factor]
                    )

        item_vectors[item_index, :] = np.linalg.solve(system, target)


@njit(parallel=True, cache=True)
def _squared_error(
    indptr: np.ndarray,
    item_indices: np.ndarray,
    ratings: np.ndarray,
    user_biases: np.ndarray,
    item_biases: np.ndarray,
    user_vectors: np.ndarray,
    item_vectors: np.ndarray,
) -> tuple[float, int]:
    squared_error = 0.0
    rating_count = 0
    latent_dim = user_vectors.shape[1]

    for user_index in prange(user_biases.size):
        user_squared_error = 0.0
        user_rating_count = 0
        for rating_index in range(indptr[user_index], indptr[user_index + 1]):
            item_index = item_indices[rating_index]
            prediction = user_biases[user_index] + item_biases[item_index]
            for factor_index in range(latent_dim):
                prediction += user_vectors[user_index, factor_index] * item_vectors[item_index, factor_index]
            residual = ratings[rating_index] - prediction
            user_squared_error += residual * residual
            user_rating_count += 1
        squared_error += user_squared_error
        rating_count += user_rating_count

    return squared_error, rating_count


def _evaluate(
    matrix,
    user_biases: np.ndarray,
    item_biases: np.ndarray,
    user_vectors: np.ndarray,
    item_vectors: np.ndarray,
    active_users: np.ndarray,
    active_items: np.ndarray,
    lamb: float,
    gamma: float,
    tau: float,
) -> tuple[float, float]:
    squared_error, count = _squared_error(
        matrix.indptr,
        matrix.indices,
        matrix.data,
        user_biases,
        item_biases,
        user_vectors,
        item_vectors,
    )
    rmse = float(np.sqrt(squared_error / count)) if count else float("nan")
    regularization = gamma * (
        np.sum(user_biases[active_users] ** 2) + np.sum(item_biases[active_items] ** 2)
    )
    regularization += tau * (
        np.sum(user_vectors[active_users] ** 2) + np.sum(item_vectors[active_items] ** 2)
    )
    objective = 0.5 * lamb * squared_error + 0.5 * regularization
    return rmse, float(objective)


def fit_numba_als(
    data: SparseRatings,
    epochs: int,
    latent_dim: int,
    lamb: float,
    gamma: float,
    tau: float,
    random_state: int = 42,
    threads: int | None = None,
) -> ALSResult:
    if epochs < 1 or latent_dim < 1:
        raise ValueError("epochs and latent_dim must be positive")
    if lamb <= 0 or gamma < 0 or tau <= 0:
        raise ValueError("lamb and tau must be positive, and gamma must be non-negative")

    thread_count = numba.get_num_threads() if threads is None else threads
    if thread_count < 1 or thread_count > numba.config.NUMBA_NUM_THREADS:
        raise ValueError(f"threads must be between 1 and {numba.config.NUMBA_NUM_THREADS}")

    previous_thread_count = numba.get_num_threads()
    numba.set_num_threads(thread_count)
    try:
        rng = np.random.default_rng(random_state)
        user_biases = np.zeros(len(data.user_ids), dtype=np.float64)
        item_biases = np.zeros(len(data.item_ids), dtype=np.float64)
        user_vectors = rng.normal(0.0, 0.1, (len(data.user_ids), latent_dim))
        item_vectors = rng.normal(0.0, 0.1, (len(data.item_ids), latent_dim))

        train_active_users = np.flatnonzero(np.diff(data.train_by_user.indptr))
        train_active_items = np.arange(len(data.item_ids))
        test_active_users = np.flatnonzero(np.diff(data.test_by_user.indptr))
        test_active_items = np.unique(data.test_by_user.indices)
        losses: list[float] = []
        losses_test: list[float] = []
        train_rmse: list[float] = []
        test_rmse: list[float] = []

        for _ in range(epochs):
            _update_users(
                data.train_by_user.indptr,
                data.train_by_user.indices,
                data.train_by_user.data,
                user_biases,
                item_biases,
                user_vectors,
                item_vectors,
                lamb,
                gamma,
                tau,
            )
            _update_items(
                data.train_by_item.indptr,
                data.train_by_item.indices,
                data.train_by_item.data,
                user_biases,
                item_biases,
                user_vectors,
                item_vectors,
                lamb,
                gamma,
                tau,
            )

            current_train_rmse, train_loss = _evaluate(
                data.train_by_user,
                user_biases,
                item_biases,
                user_vectors,
                item_vectors,
                train_active_users,
                train_active_items,
                lamb,
                gamma,
                tau,
            )
            current_test_rmse, test_loss = _evaluate(
                data.test_by_user,
                user_biases,
                item_biases,
                user_vectors,
                item_vectors,
                test_active_users,
                test_active_items,
                lamb,
                gamma,
                tau,
            )
            train_rmse.append(current_train_rmse)
            test_rmse.append(current_test_rmse)
            losses.append(train_loss)
            losses_test.append(test_loss)

        return ALSResult(
            user_biases=user_biases,
            item_biases=item_biases,
            user_vectors=user_vectors,
            item_vectors=item_vectors,
            losses=losses,
            losses_test=losses_test,
            train_rmse=train_rmse,
            test_rmse=test_rmse,
        )
    finally:
        numba.set_num_threads(previous_thread_count)