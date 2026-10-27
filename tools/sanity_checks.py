"""Numerical checks of the closed-form updates and claims in the report.

Runs on a small synthetic dataset (no downloads needed):

    python tools/make_synthetic_movielens.py --out datasets/ml-synthetic
    python tools/sanity_checks.py --data-dir datasets/ml-synthetic
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from recsys_models.als import (  # noqa: E402
    ALSConfig, _update_genre_vecs, _update_side, fit_als, item_scale_matrix,
    objective)
from recsys_models.bpr import BPRConfig, fit_bpr, positives_csr  # noqa: E402
from recsys_models.data import (  # noqa: E402
    by_user, load_movielens, random_split)
from recsys_models.vb import VBConfig, fit_vb  # noqa: E402

RESULTS = []


def check(name: str, passed: bool, detail: str = "") -> None:
    RESULTS.append(passed)
    print(f"[{'PASS' if passed else 'FAIL'}] {name} {detail}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    args = parser.parse_args()
    data = load_movielens(args.data_dir)
    train, val, test = random_split(data.ratings, data.n_users, data.n_items)
    m, n = data.n_users, data.n_items

    # 1. every ALS variant has a non-increasing objective -------------------
    for name, k, genre in (("biases only", 0, None), ("MF", 6, None),
                           ("MF + genres", 6, data.genre_matrix)):
        model = fit_als(train, m, n, ALSConfig(latent_dim=k, epochs=12),
                        genre)
        steps = np.diff(model.history["objective"])
        check(f"ALS objective non-increasing ({name})",
              bool(np.all(steps <= 1e-8)), f"max step {steps.max():.2e}")

    # 2. each user update zeroes the gradient of the objective -------------
    cfg = ALSConfig(latent_dim=4, epochs=3)
    model = fit_als(train, m, n, cfg)
    _update_side(by_user(train, m, n), model.user_bias, model.user_vecs,
                 model.item_bias, model.item_vecs, cfg, None)
    row = 7
    base = objective(model, train)
    worst = 0.0
    # The bias is updated before the vector (block coordinate descent), so
    # after the step only the vector block is stationary.
    for idx in range(cfg.latent_dim):
        arr = model.user_vecs[row]
        old = arr[idx]
        arr[idx] = old + 1e-6
        up = objective(model, train)
        arr[idx] = old - 1e-6
        down = objective(model, train)
        arr[idx] = old
        worst = max(worst, abs((up - down) / 2e-6))
    check("user vector update is stationary (finite differences)",
          worst < 1e-4, f"max |gradient| {worst:.1e}, objective {base:.1f}")

    # 3. genre-vector update is jointly stationary -----------------------
    cfg = ALSConfig(latent_dim=3, epochs=4)
    model = fit_als(train, m, n, cfg, data.genre_matrix)
    scale = item_scale_matrix(data.genre_matrix)
    _update_genre_vecs(model.genre_vecs, model.item_vecs, scale, cfg)
    # Analytic gradient of the objective w.r.t. F must vanish after the step.
    gradient = (-cfg.tau * scale.T @ (model.item_vecs - scale
                                      @ model.genre_vecs)
                + cfg.tau_feat * model.genre_vecs)
    check("genre-vector update is jointly stationary",
          float(np.abs(gradient).max()) < 1e-9,
          f"max |gradient| {np.abs(gradient).max():.1e}")

    # 4. MAP solution is invariant to a joint rescaling of lambda, tau, gamma
    a = fit_als(train, m, n, ALSConfig(latent_dim=5, lamb=0.2, gamma=0.1,
                                       tau=0.7, epochs=15))
    b = fit_als(train, m, n, ALSConfig(latent_dim=5, lamb=2.0, gamma=1.0,
                                       tau=7.0, epochs=15))
    gap = float(np.max(np.abs(a.user_vecs - b.user_vecs)))
    check("MAP invariant to scaling (lambda, tau, gamma) jointly by 10",
          gap < 1e-8, f"max parameter difference {gap:.1e}")

    # 5. BPR gradient direction -----------------------------------------------
    rng = np.random.default_rng(0)
    u, vi, vj = rng.normal(0, .5, (3, 5))
    bi, bj, reg = 0.3, -0.2, 0.1

    def objective_bpr(u_, vi_, vj_):
        x = u_ @ (vi_ - vj_) + bi - bj
        return -np.logaddexp(0, -x) - reg / 2 * (u_ @ u_ + vi_ @ vi_
                                                 + vj_ @ vj_)

    x = u @ (vi - vj) + bi - bj
    w = 1.0 / (1.0 + np.exp(x))
    analytic = w * (vi - vj) - reg * u
    numeric = np.array([(objective_bpr(u + 1e-6 * e, vi, vj)
                         - objective_bpr(u - 1e-6 * e, vi, vj)) / 2e-6
                        for e in np.eye(5)])
    check("BPR user gradient matches finite differences",
          bool(np.allclose(analytic, numeric, atol=1e-7)))

    # 6. BPR separates planted clusters -----------------------------------------
    from recsys_models.data import Ratings
    from recsys_models.metrics import ranking_metrics
    users_, items_ = 120, 160
    uc = rng.integers(0, 4, users_)
    ic = rng.integers(0, 4, items_)
    prob = np.where(uc[:, None] == ic[None, :], 0.3, 0.02)
    uu, ii = np.nonzero(rng.random((users_, items_)) < prob)
    pos = Ratings(uu.astype(np.int32), ii.astype(np.int32),
                  np.ones(len(uu)))
    hold = rng.random(len(uu)) < 0.2
    tr, te = pos.subset(~hold), pos.subset(hold)
    bpr = fit_bpr(tr, users_, items_, BPRConfig(epochs=40, latent_dim=8))
    metrics = ranking_metrics(bpr.score_matrix(),
                              positives_csr(tr, users_, items_),
                              positives_csr(te, users_, items_), (10,))
    check("BPR beats chance on planted clusters (MPR < 0.3)",
          metrics["mpr"] < 0.3, f"MPR {metrics['mpr']:.3f}")

    # 7. VB: ELBO increases monotonically -------------------------------------
    vb = fit_vb(train, m, n, VBConfig(latent_dim=4, lamb=1.5, tau=4.0,
                                      epochs=10))
    steps = np.diff(vb.history["elbo"])
    check("VB ELBO non-decreasing", bool(np.all(steps >= -1e-6)),
          f"min step {steps.min():.2e}")

    print(f"\n{sum(RESULTS)}/{len(RESULTS)} checks passed")
    sys.exit(0 if all(RESULTS) else 1)


if __name__ == "__main__":
    main()
