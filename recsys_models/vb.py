"""Mean-field variational Bayes for biased matrix factorisation.

Generative model (same hyper-parameters as the ALS objective)::

    r_mn | .  ~ N(u_m . v_n + b_m + c_n, 1 / lamb)
    u_m, v_n  ~ N(0, I / tau)
    b_m, c_n  ~ N(0, 1 / gamma)

Instead of the point estimates of ALS we fit a fully factorised Gaussian
posterior ``q = prod q(u_m) q(v_n) q(b_m) q(c_n)`` by coordinate ascent on
the evidence lower bound (CAVI). Every update is the exact optimum of one
factor, so the ELBO increases monotonically. Compared with ALS, the normal
equations of one side contain the posterior *covariances* of the other side,
which shrinks rows with few ratings more strongly and yields uncertainty.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from recsys_models.data import Ratings, by_item, by_user


@dataclass
class VBConfig:
    latent_dim: int = 10
    lamb: float = 1.0
    gamma: float = 1.0
    tau: float = 1.0
    epochs: int = 30
    seed: int = 42
    init_std: float = 0.1


@dataclass
class VBModel:
    """Posterior means and (co)variances of all latent quantities."""

    config: VBConfig
    user_mean: np.ndarray    # (M, K)
    user_cov: np.ndarray     # (M, K, K)
    item_mean: np.ndarray    # (N, K)
    item_cov: np.ndarray     # (N, K, K)
    user_bias_mean: np.ndarray
    user_bias_var: np.ndarray
    item_bias_mean: np.ndarray
    item_bias_var: np.ndarray
    history: dict[str, list[float]] = field(default_factory=dict)

    def _interaction_variance(self, users, items) -> np.ndarray:
        """``Var(u . v)`` under the factorised posterior."""
        su, sv = self.user_cov[users], self.item_cov[items]
        mu, mv = self.user_mean[users], self.item_mean[items]
        return (np.einsum("cij,cji->c", su, sv)
                + np.einsum("ci,cij,cj->c", mu, sv, mu)
                + np.einsum("ci,cij,cj->c", mv, su, mv))

    def predict(self, users, items, chunk: int = 4000):
        """Posterior predictive mean and variance of the ratings."""
        mean = (np.einsum("rk,rk->r", self.user_mean[users],
                          self.item_mean[items])
                + self.user_bias_mean[users] + self.item_bias_mean[items])
        variance = np.empty(len(mean))
        for start in range(0, len(mean), chunk):
            sl = slice(start, start + chunk)
            variance[sl] = (
                1.0 / self.config.lamb
                + self._interaction_variance(users[sl], items[sl])
                + self.user_bias_var[users[sl]]
                + self.item_bias_var[items[sl]])
        return mean, variance

    def rmse(self, data: Ratings) -> float:
        mean, _ = self.predict(data.users, data.items)
        return float(np.sqrt(np.mean((data.values - mean) ** 2)))

    def predictive_scores(self, data: Ratings) -> dict[str, float]:
        """RMSE, mean predictive log-density and 90% interval coverage."""
        mean, var = self.predict(data.users, data.items)
        err = data.values - mean
        log_density = -0.5 * np.log(2 * np.pi * var) - err ** 2 / (2 * var)
        covered = np.abs(err) <= 1.6449 * np.sqrt(var)
        return {"rmse": float(np.sqrt(np.mean(err ** 2))),
                "log_density": float(np.mean(log_density)),
                "coverage90": float(np.mean(covered))}


def _update_side(csr, mean, cov, bias_mean, bias_var, other_mean, other_cov,
                 other_bias_mean, cfg: VBConfig) -> None:
    """CAVI update of every row of ``csr`` (bias first, then trait vector)."""
    k = mean.shape[1]
    prior_precision = cfg.tau * np.eye(k)
    for row in range(csr.shape[0]):
        start, stop = csr.indptr[row], csr.indptr[row + 1]
        count = stop - start
        if count == 0:
            bias_mean[row], bias_var[row] = 0.0, 1.0 / cfg.gamma
            mean[row], cov[row] = 0.0, np.eye(k) / cfg.tau
            continue
        cols = csr.indices[start:stop]
        values = csr.data[start:stop]
        partner = other_mean[cols]
        ob = other_bias_mean[cols]

        precision_b = cfg.gamma + cfg.lamb * count
        bias_mean[row] = cfg.lamb * np.sum(
            values - partner @ mean[row] - ob) / precision_b
        bias_var[row] = 1.0 / precision_b

        second_moment = partner.T @ partner + other_cov[cols].sum(axis=0)
        precision = prior_precision + cfg.lamb * second_moment
        cov[row] = np.linalg.inv(precision)
        cov[row] = 0.5 * (cov[row] + cov[row].T)
        mean[row] = cov[row] @ (
            cfg.lamb * (partner.T @ (values - bias_mean[row] - ob)))


def elbo(model: VBModel, train: Ratings, chunk: int = 4000) -> float:
    """Evidence lower bound on the training ratings (up to no constants)."""
    cfg = model.config
    k = cfg.latent_dim
    mean = (np.einsum("rk,rk->r", model.user_mean[train.users],
                      model.item_mean[train.items])
            + model.user_bias_mean[train.users]
            + model.item_bias_mean[train.items])
    expected_sq = (train.values - mean) ** 2
    expected_sq += model.user_bias_var[train.users]
    expected_sq += model.item_bias_var[train.items]
    for start in range(0, len(train), chunk):
        sl = slice(start, start + chunk)
        expected_sq[sl] += model._interaction_variance(
            train.users[sl], train.items[sl])
    log_lik = np.sum(0.5 * np.log(cfg.lamb / (2 * np.pi))
                     - 0.5 * cfg.lamb * expected_sq)

    def kl_vectors(means, covs):
        trace = np.trace(covs, axis1=1, axis2=2)
        _, logdet = np.linalg.slogdet(cfg.tau * covs)
        return 0.5 * np.sum(cfg.tau * trace + cfg.tau * np.sum(means ** 2, 1)
                            - k - logdet)

    def kl_scalars(means, variances):
        return 0.5 * np.sum(cfg.gamma * variances + cfg.gamma * means ** 2
                            - 1.0 - np.log(cfg.gamma * variances))

    kl = (kl_vectors(model.user_mean, model.user_cov)
          + kl_vectors(model.item_mean, model.item_cov)
          + kl_scalars(model.user_bias_mean, model.user_bias_var)
          + kl_scalars(model.item_bias_mean, model.item_bias_var))
    return float(log_lik - kl)


def fit_vb(
    train: Ratings,
    n_users: int,
    n_items: int,
    cfg: VBConfig,
    eval_sets: dict[str, Ratings] | None = None,
    verbose: bool = False,
) -> VBModel:
    """Run CAVI and record the ELBO and RMSEs after every epoch."""
    rng = np.random.default_rng(cfg.seed)
    k = cfg.latent_dim
    model = VBModel(
        config=cfg,
        user_mean=rng.normal(0.0, cfg.init_std, (n_users, k)),
        user_cov=np.tile(np.eye(k) / cfg.tau, (n_users, 1, 1)),
        item_mean=rng.normal(0.0, cfg.init_std, (n_items, k)),
        item_cov=np.tile(np.eye(k) / cfg.tau, (n_items, 1, 1)),
        user_bias_mean=np.zeros(n_users),
        user_bias_var=np.full(n_users, 1.0 / cfg.gamma),
        item_bias_mean=np.zeros(n_items),
        item_bias_var=np.full(n_items, 1.0 / cfg.gamma),
    )
    csr_user = by_user(train, n_users, n_items)
    csr_item = by_item(train, n_users, n_items)
    eval_sets = eval_sets or {}
    history: dict[str, list[float]] = {"elbo": [], "train_rmse": []}
    for name in eval_sets:
        history[f"{name}_rmse"] = []

    for epoch in range(cfg.epochs):
        _update_side(csr_user, model.user_mean, model.user_cov,
                     model.user_bias_mean, model.user_bias_var,
                     model.item_mean, model.item_cov,
                     model.item_bias_mean, cfg)
        _update_side(csr_item, model.item_mean, model.item_cov,
                     model.item_bias_mean, model.item_bias_var,
                     model.user_mean, model.user_cov,
                     model.user_bias_mean, cfg)
        history["elbo"].append(elbo(model, train))
        history["train_rmse"].append(model.rmse(train))
        for name, data in eval_sets.items():
            history[f"{name}_rmse"].append(model.rmse(data))
        if verbose:
            print(f"epoch {epoch + 1:3d}  ELBO {history['elbo'][-1]:.1f}  "
                  f"train RMSE {history['train_rmse'][-1]:.4f}")
    model.history = history
    return model
