"""Turn experiment outputs into the inputs expected by the LaTeX report.

Reads ``output/<experiment>/results.json`` and the saved plots and writes

* ``<report>/figures/*.png``      copies of the plots under stable names,
* ``<report>/generated/results.tex``  ``\\setres{key}{value}`` macros,
* ``<report>/generated/<table>.tex``  ready-made LaTeX tables.

Anything whose source is missing is skipped, and the report then shows a red
placeholder for it. Run it again after every experiment:

    python tools/make_report_assets.py
"""
import argparse
import json
import re
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "report"

# destination name -> (experiment directory, glob pattern, exclude substring)
FIGURES = {
    "p01_rating.png": ("practice_01", "rating_distribution.png", None),
    "p01_degree.png": ("practice_01", "degree_distribution.png", None),
    "ds_ccdf.png": ("dataset_analysis", "degree_ccdf.png", None),
    "p02_obj.png": ("practice_02/plots",
                    "*Regularized_Negative_Log_Likelihood_objective_*.png",
                    "(Test)"),
    "p03_obj.png": ("practice_03/plots",
                    "*Regularized_Negative_Log_Likelihood_objective_*.png",
                    "(Test)"),
    "p03_rmse.png": ("practice_03/plots",
                     "*Root_Mean_Squared_Error_metrics_*.png", None),
    "p04_obj.png": ("practice_04/plots",
                    "*Regularized_Negative_Log_Likelihood_objective_*.png",
                    "(Test)"),
    "p05_emb_plain.png": ("practice_05/plots", "item_embedding_plain.png",
                          None),
    "p05_emb_genre.png": ("practice_05/plots", "item_embedding_genre.png",
                          None),
    "p05_genre_vectors.png": ("practice_05/plots", "genre_vectors.png", None),
    "p06_precision.png": ("practice_06/plots", "precision_at_k.png", None),
    "p06_recall.png": ("practice_06/plots", "recall_at_k.png", None),
    "p07_ctr.png": ("practice_07/plots", "ctr_comparison.png", None),
    "p07_pab.png": ("practice_07/plots", "pvalues_ab.png", None),
    "p07_paa.png": ("practice_07/plots", "pvalues_aa.png", None),
    "p08_heatmap.png": ("practice_08/plots", "lambda_tau_heatmap.png", None),
    "p08_ksweep.png": ("practice_08/plots", "k_sweep.png", None),
    "p08_gamma.png": ("practice_08/plots", "gamma_sweep.png", None),
    "p09_normstd.png": ("practice_09/plots", "norm_vs_rating_std.png", None),
    "p10_elbo.png": ("practice_10/plots", "elbo.png", None),
    "p10_uncertainty.png": ("practice_10/plots", "bias_uncertainty.png",
                            None),
    "bench_speedup.png": ("benchmark/plots", "speedup.png", None),
}


def escape(text: str) -> str:
    """Escape LaTeX special characters in movie titles."""
    table = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$",
             "#": r"\#", "_": r"\_", "{": r"\{", "}": r"\}",
             "~": r"\textasciitilde{}", "^": r"\textasciicircum{}"}
    return "".join(table.get(c, c) for c in text)


def bin_label(label: str) -> str:
    """Render ``>=26`` as a math symbol (a bare ``>`` breaks in text mode)."""
    return label.replace(">=", "$\\geq$")


def num(value: float, digits: int = 4) -> str:
    """Fixed-point number with a typographic minus (works in any mode)."""
    text = f"{abs(value):.{digits}f}"
    return "\\ensuremath{-}" + text if value < 0 and float(text) else text


def grouped(value: int) -> str:
    return f"{int(value):,}".replace(",", "{,}")


def pvalue(value: float) -> str:
    """Math-mode safe p-value."""
    if value < 1e-3:
        mantissa, exponent = f"{value:.1e}".split("e")
        return f"{mantissa}\\times10^{{{int(exponent)}}}"
    return f"{value:.4f}"


def percent(value: float) -> str:
    text = f"{100 * abs(value):.1f}\\%"
    return "\\ensuremath{-}" + text if value < 0 else text


def load(output: Path, name: str) -> dict | None:
    path = output / name / "results.json"
    if not path.exists():
        print(f"  - missing {path.relative_to(ROOT)}")
        return None
    return json.loads(path.read_text())


def best_of(values: list[float], best: str = "max") -> int:
    return values.index(max(values) if best == "max" else min(values))


def tabular(spec: str, header: list[str], rows: list[list[str]]) -> str:
    lines = [f"\\begin{{tabular}}{{{spec}}}", "\\toprule",
             " & ".join(header) + " \\\\", "\\midrule"]
    lines += [" & ".join(row) + " \\\\" for row in rows]
    lines += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(lines) + "\n"


def build(output: Path) -> tuple[dict[str, str], dict[str, str]]:
    macros: dict[str, str] = {}
    tables: dict[str, str] = {}

    d = load(output, "dataset_analysis")
    if d:
        macros.update(
            nRatings=grouped(d["n_ratings"]), nUsers=grouped(d["n_users"]),
            nItems=grouped(d["n_items"]), nGenres=str(d["n_genres"]),
            ratingMean=num(d["rating_mean"], 2),
            ratingStd=num(d["rating_std"], 2),
            itemMedian=num(d["item_degree"]["median"], 0),
            itemMax=grouped(d["item_degree"]["max"]),
            itemGini=num(d["item_degree"]["gini"], 2),
            alphaItem=num(d["power_law"]["items"]["alpha"], 2),
            xminItem=num(d["power_law"]["items"]["x_min"], 0),
            alphaUser=num(d["power_law"]["users"]["alpha"], 2),
            xminUser=num(d["power_law"]["users"]["x_min"], 0))

    t = load(output, "practice_08")
    if t:
        final, bias = t["final"], t["biases_only"]
        macros.update(
            nTrain=grouped(t["split"]["train"]),
            nVal=grouped(t["split"]["val"]),
            nTest=grouped(t["split"]["test"]),
            tuneLamb=f"{t['best_lamb']:g}", tuneTau=f"{t['best_tau_grid']:g}",
            tuneGridBest=num(t["best_grid_val_rmse"]),
            tuneGridWorst=num(t["worst_grid_val_rmse"]),
            tuneBestGamma=f"{t['best_gamma']:g}",
            tuneBiasVal=num(bias["val_rmse"]),
            tuneBiasTest=num(bias["test_rmse"]),
            tuneK=str(t["best_k"]), tuneKTau=f"{t['best_k_tau']:g}",
            tuneTrain=num(final["train_rmse"]), tuneVal=num(final["val_rmse"]),
            tuneTest=num(final["test_rmse"]))
        val = [r["val_rmse"] for r in t["k_rows"]]
        rows = []
        for i, r in enumerate(t["k_rows"]):
            cell = num(r["val_rmse"])
            rows.append([str(r["k"]), f"{r['best_tau']:g}",
                         num(r["train_rmse"]),
                         f"\\textbf{{{cell}}}" if i == best_of(val, "min")
                         else cell, num(r["val_rmse_at_global_tau"])])
        tables["kTable"] = tabular(
            "ccccc", ["$K$", "best $\\tau$", "train", "val",
                      "val ($\\tau$ fixed)"], rows)

    f = load(output, "practice_05")
    if f:
        macros.update(
            k2Tau=f"{f['hyperparameters']['tau']:g}",
            k2PlainTest=num(f["k2"]["plain"]["test_rmse"]),
            k2GenreTest=num(f["k2"]["genre"]["test_rmse"]),
            coldDim=str(f["cold_start_dim"]),
            coldPlainTest=num(f["cold_start_full_model"]["plain"]
                              ["test_rmse"]),
            coldGenreTest=num(f["cold_start_full_model"]["genre"]
                              ["test_rmse"]))
        rows = [[bin_label(b["bin"]), grouped(b["n_test"]),
                 num(b["rmse_plain"]),
                 num(b["rmse_genre"]),
                 num(b["rmse_plain"] - b["rmse_genre"])]
                for b in f["cold_start_buckets"]]
        tables["coldTable"] = tabular(
            "lcccc", ["train ratings", "test n", "no genres", "genres",
                      "gain"], rows)
        rows = [[escape(anchor), escape("; ".join(v["similar"])),
                 escape("; ".join(v["opposite"]))]
                for anchor, v in f["neighbours"].items()]
        if rows:
            tables["neighbours"] = (
                "\\begin{scriptsize}\n" + tabular(
                    "p{0.2\\columnwidth}p{0.36\\columnwidth}"
                    "p{0.36\\columnwidth}",
                    ["anchor", "most similar", "most opposite"], rows)
                + "\\end{scriptsize}\n")

    q = load(output, "practice_09")
    if q:
        pol = q["polarization"]
        macros.update(polarMinRatings=str(q.get("min_ratings", 50)),
                      polarSpearman=num(pol["spearman_norm_vs_std"], 2),
                      polarEligible=grouped(pol["eligible_movies"]))
        profile = q["profiles"].get("lord_of_the_rings")
        if profile:
            key = [k for k in profile if k.startswith("min")][0]
            rows = []
            for i, (a, b) in enumerate(zip(profile["unfiltered"],
                                           profile[key]), 1):
                rows.append([str(i),
                             f"{escape(a['title'])} ({a['n_ratings']})",
                             f"{escape(b['title'])} ({b['n_ratings']})"])
            tables["dummyLotr"] = tabular(
                "cp{0.38\\columnwidth}p{0.38\\columnwidth}",
                ["\\#", "no filter (\\#ratings)",
                 f"$\\geq${key[3:]} ratings"], rows)
        for name, label in (("most", "Most polarising"),
                            ("least", "Least polarising")):
            rows = [[escape(r["title"]), num(r["norm"], 2),
                     str(r["count"]), num(r["std"], 2)] for r in pol[name]]
            tables["polar" + name.capitalize()] = tabular(
                "p{0.5\\columnwidth}ccc",
                [label, "$\\lVert v_n\\rVert$", "\\#ratings", "std"], rows)

    b = load(output, "practice_06")
    if b:
        macros.update(
            bprThreshold=f"{b['threshold']:g}",
            bprPosTrain=grouped(b["n_train_pos"]),
            bprPosTest=grouped(b["n_test_pos"]),
            bprReg=f"{b['chosen_reg']:g}", bprEpochs=str(b["chosen_epochs"]),
            bprLossFirst=num(b["bpr_loss_first"], 3),
            bprLossLast=num(b["bpr_loss_last"], 3))
        names = [("random", "Random"), ("popularity", "Popularity"),
                 ("explicit_als", "Explicit ALS"), ("bpr", "BPR")]
        columns = ["precision@10", "recall@10", "ndcg@10", "mpr"]
        best = {c: (min if c == "mpr" else max)(
            b["test"][k][c] for k, _ in names) for c in columns}
        rows = []
        for key, label in names:
            cells = []
            for c in columns:
                text = num(b["test"][key][c])
                cells.append(f"\\textbf{{{text}}}"
                             if b["test"][key][c] == best[c] else text)
            rows.append([label] + cells)
        tables["bprTable"] = tabular(
            "lcccc", ["Model", "P@10", "R@10", "NDCG@10", "MPR"], rows)

    a = load(output, "practice_07")
    if a:
        ab, aa = a["ab_test"], a["aa_test"]
        macros.update(
            abSessions=str(a["sessions"]), abNrecs=str(a["n_recs"]),
            abRepeats=str(a["repeats"]),
            clickSlope=f"{a['click_model']['slope']:g}",
            clickCentre=f"{a['click_model']['centre']:g}",
            abUsersA=grouped(a["n_users_a"]), abUsersB=grouped(a["n_users_b"]),
            abLogged=grouped(a["n_logged_interactions"]),
            abCtrA=num(ab["ctr_control"]), abCtrB=num(ab["ctr_treatment"]),
            abDiff=num(ab["difference"]), abCiLow=num(ab["ci95_low"]),
            abCiHigh=num(ab["ci95_high"]), abLift=percent(
                ab["relative_lift"]),
            abT=num(ab["t_statistic"], 2), abP=pvalue(ab["p_value"]),
            abD=num(ab["cohens_d"], 2), aaP=pvalue(aa["p_value"]),
            abPower=num(a["power_ab"], 3), aaFpr=num(a["false_positive_aa"], 3),
            abOverlap=num(a["mean_top10_overlap_a_b"], 2))
        rows = [[v, num(m["ctr"]), num(m["mean_log_popularity"], 2),
                 grouped(m["distinct_items"]),
                 percent(m["catalog_coverage"]),
                 num(m["mean_true_rating"], 2)]
                for v, m in a["behaviour"].items()]
        tables["abBehaviour"] = tabular(
            "lccccc", ["version", "CTR", "mean log(1+pop.)", "distinct",
                       "coverage", "mean true rating"], rows)

    v = load(output, "practice_10")
    if v:
        c = v["config"]
        macros.update(
            vbK=str(c["latent_dim"]), vbLamb=f"{c['lamb']:g}",
            vbTau=f"{c['tau']:.3g}", vbGamma=f"{c['gamma']:.3g}",
            vbMapTest=num(v["map"]["rmse"]), vbVbTest=num(v["vb"]["rmse"]),
            vbLogDens=num(v["vb"]["log_density"], 3),
            mapLogDens=num(v["map"]["log_density"], 3),
            vbCov=percent(v["vb"]["coverage90"]),
            mapCov=percent(v["map"]["coverage90"]))
        rows = [[bin_label(x["bin"]), grouped(x["n_test"]),
                 num(x["rmse_map"]),
                 num(x["rmse_vb"]), num(x["mean_pred_std"], 2)]
                for x in v["buckets"]]
        tables["vbBuckets"] = tabular(
            "lcccc", ["train ratings", "test n", "MAP", "VB",
                      "mean pred.\\ std"], rows)

    k = load(output, "benchmark")
    if k:
        threads = sorted(int(x) for x in k["seconds_per_epoch"])
        best_t = max(threads, key=lambda x: k["speedup"][str(x)])
        macros.update(benchCpus=str(k["cpu_count"]),
                      benchThreads=str(best_t),
                      benchSpeedup=f"{k['speedup'][str(best_t)]:.1f}$\\times$")
        rows = [[str(t), num(k["seconds_per_epoch"][str(t)], 2),
                 num(k["speedup"][str(t)], 2),
                 percent(k["efficiency"][str(t)])] for t in threads]
        tables["benchTable"] = tabular(
            "cccc", ["threads", "s / iteration", "speed-up", "efficiency"],
            rows)
    return macros, tables


def copy_figures(output: Path, figures: Path) -> None:
    figures.mkdir(parents=True, exist_ok=True)
    for destination, (folder, pattern, exclude) in FIGURES.items():
        matches = sorted(p for p in (output / folder).glob(pattern)
                         if exclude is None or exclude not in p.name)
        if matches:
            shutil.copy(matches[0], figures / destination)
        else:
            print(f"  - missing figure {destination} "
                  f"({folder}/{pattern})")


def report_missing(report: Path, macros: dict[str, str]) -> None:
    """List \\res{} keys of main.tex that have neither default nor value."""
    text = (report / "main.tex").read_text()
    used = set(re.findall(r"\\res\{([A-Za-z0-9]+)\}", text)) - {"key"}
    defaults = set(re.findall(r"\\setres\{([A-Za-z0-9]+)\}", text))
    missing = sorted(used - defaults - set(macros))
    if missing:
        print("\nStill missing values (red in the PDF): " + ", ".join(missing))
    else:
        print("\nAll numeric values are available.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "output")
    parser.add_argument("--report-dir", type=Path, default=REPORT)
    args = parser.parse_args()

    print("Reading experiment results:")
    macros, tables = build(args.output_dir)
    generated = args.report_dir / "generated"
    generated.mkdir(parents=True, exist_ok=True)
    lines = [f"\\setres{{{key}}}{{{value}}}"
             for key, value in sorted(macros.items())]
    (generated / "results.tex").write_text("\n".join(lines) + "\n")
    for name, content in tables.items():
        (generated / f"{name}.tex").write_text(content)
    print(f"\nWrote {len(macros)} values and {len(tables)} tables.")
    print("Copying figures:")
    copy_figures(args.output_dir, args.report_dir / "figures")
    report_missing(args.report_dir, macros)


if __name__ == "__main__":
    main()
