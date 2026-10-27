"""Practical 6: implicit feedback and Bayesian Personalized Ranking (BPR).

Ratings >= ``--threshold`` are treated as positive implicit feedback. For
every user 10% / 20% of the positives are held out for validation / test.
Compared on Precision@k, Recall@k, NDCG@k and mean percentile rank:

* random scores,
* item popularity (number of training positives),
* the explicit ALS model (``u . v + b_i``), trained on every rating except
  the held-out positives,
* BPR, with its regularisation and number of epochs picked on validation.

Usage:
    python practice_06.py --data-dir datasets/ml-latest-small
"""
import argparse
import json
from pathlib import Path

import numpy as np

from recsys_models.als import ALSConfig, fit_als
from recsys_models.bpr import (
    BPRConfig, fit_bpr, popularity_scores, positives_csr, positives_of,
    split_positives)
from recsys_models.data import load_movielens
from recsys_models.metrics import ranking_metrics
from visulizations.plots import plot_lines

REGS = (0.003, 0.01, 0.03, 0.1)
EVAL_EVERY = 5
KS = (1, 2, 5, 10, 20, 30, 40, 50)


def main() -> None:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path,
                        default=root / "datasets" / "ml-latest-small")
    parser.add_argument("--output-dir", type=Path,
                        default=root / "output" / "practice_06")
    parser.add_argument("--threshold", type=float, default=4.0)
    parser.add_argument("--latent-dim", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    regs = (0.01, 0.1) if args.quick else REGS
    epochs = 10 if args.quick else args.epochs

    data = load_movielens(args.data_dir)
    n_u, n_i = data.n_users, data.n_items
    ratings = data.ratings
    label = split_positives(ratings, n_u, args.threshold, seed=args.seed)
    train_pos, val_pos, test_pos = (positives_of(ratings, label, k)
                                    for k in (0, 1, 2))
    train_c = positives_csr(train_pos, n_u, n_i)
    val_c = positives_csr(val_pos, n_u, n_i)
    test_c = positives_csr(test_pos, n_u, n_i)
    print(f"positives: train {len(train_pos)}, val {len(val_pos)}, "
          f"test {len(test_pos)}")

    def evaluate(scores, target, ks=(5, 10, 20)):
        return ranking_metrics(scores, train_c, target, ks)

    # --- baselines ------------------------------------------------------
    rng = np.random.default_rng(args.seed)
    results = {
        "random": evaluate(rng.random((n_u, n_i)), test_c),
        "popularity": evaluate(popularity_scores(train_c, n_u, n_i), test_c),
    }
    explicit_train = ratings.subset((label == 0) | (label == 3))
    explicit = fit_als(explicit_train, n_u, n_i,
                       ALSConfig(latent_dim=10, lamb=0.09, gamma=0.1,
                                 tau=0.9, epochs=30, seed=args.seed))
    explicit_scores = (explicit.user_vecs @ explicit.item_vecs.T
                       + explicit.item_bias[None, :])
    results["explicit_als"] = evaluate(explicit_scores, test_c)

    # --- BPR model selection on validation ---------------------------------
    search = []
    for reg in regs:
        cfg = BPRConfig(latent_dim=args.latent_dim, learning_rate=args.lr,
                        reg=reg, reg_bias=reg, epochs=epochs, seed=args.seed)

        def on_epoch(epoch, model):
            if (epoch + 1) % EVAL_EVERY and epoch + 1 != epochs:
                return {}
            m = evaluate(model.score_matrix(), val_c, (10,))
            return {"val_recall10": m["recall@10"],
                    "val_precision10": m["precision@10"],
                    "val_ndcg10": m["ndcg@10"], "epoch": float(epoch + 1)}

        model = fit_bpr(train_pos, n_u, n_i, cfg, on_epoch=on_epoch)
        curve = model.history["val_ndcg10"]
        stops = model.history["epoch"]
        best = int(np.argmax(curve))
        search.append({"reg": reg, "best_epoch": int(stops[best]),
                       "val_ndcg10": curve[best],
                       "val_recall10": model.history["val_recall10"][best],
                       "curve_epochs": stops, "curve_ndcg10": curve,
                       "bpr_loss": model.history["bpr_loss"]})
        print(f"[BPR reg {reg:g}] best val NDCG@10 {curve[best]:.4f} at "
              f"epoch {int(stops[best])}")
    chosen = max(search, key=lambda r: r["val_ndcg10"])

    # --- final BPR (same seed => same trajectory), evaluate on test --------
    final_cfg = BPRConfig(latent_dim=args.latent_dim, learning_rate=args.lr,
                          reg=chosen["reg"], reg_bias=chosen["reg"],
                          epochs=chosen["best_epoch"], seed=args.seed)
    final = fit_bpr(train_pos, n_u, n_i, final_cfg)
    scores = final.score_matrix()
    results["bpr"] = evaluate(scores, test_c)

    # --- precision / recall as a function of k ----------------------------------
    curves = {}
    for name, sc in (("BPR", scores),
                     ("popularity", popularity_scores(train_c, n_u, n_i)),
                     ("explicit ALS", explicit_scores)):
        m = ranking_metrics(sc, train_c, test_c, KS)
        curves[name] = {"precision": [m[f"precision@{k}"] for k in KS],
                        "recall": [m[f"recall@{k}"] for k in KS]}
    plots = args.output_dir / "plots"
    plot_lines(list(KS), {n: c["precision"] for n, c in curves.items()},
               "k", "Precision@k", "Precision@k", "precision_at_k.png",
               plots)
    plot_lines(list(KS), {n: c["recall"] for n, c in curves.items()},
               "k", "Recall@k", "Recall@k", "recall_at_k.png", plots)
    full_loss = fit_bpr(train_pos, n_u, n_i, BPRConfig(
        latent_dim=args.latent_dim, learning_rate=args.lr,
        reg=chosen["reg"], reg_bias=chosen["reg"], epochs=epochs,
        seed=args.seed))
    plot_lines(list(range(1, epochs + 1)),
               {"BPR loss on fixed triplets": full_loss.history["bpr_loss"]},
               "epoch", r"mean $-\ln\sigma(\hat{x}_{uij})$",
               "BPR training loss", "bpr_loss.png", plots, markers=False)
    plot_lines(chosen["curve_epochs"],
               {f"reg={r['reg']:g}": r["curve_ndcg10"] for r in search},
               "epoch", "validation NDCG@10", "BPR model selection",
               "bpr_validation.png", plots)

    out = {
        "dataset": args.data_dir.name,
        "threshold": args.threshold,
        "n_train_pos": len(train_pos), "n_val_pos": len(val_pos),
        "n_test_pos": len(test_pos),
        "latent_dim": args.latent_dim, "lr": args.lr,
        "chosen_reg": chosen["reg"], "chosen_epochs": chosen["best_epoch"],
        "search": search, "test": results, "k_curves": curves, "ks": list(KS),
        "bpr_loss_first": full_loss.history["bpr_loss"][0],
        "bpr_loss_last": full_loss.history["bpr_loss"][-1],
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "results.json").write_text(json.dumps(out, indent=2))
    print("\nTest metrics:")
    for name, m in results.items():
        print(f"  {name:13s} P@10 {m['precision@10']:.4f}  R@10 "
              f"{m['recall@10']:.4f}  NDCG@10 {m['ndcg@10']:.4f}  MPR "
              f"{m['mpr']:.3f}")


if __name__ == "__main__":
    main()
