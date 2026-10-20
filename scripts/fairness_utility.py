"""Phase 4: fairness-utility evaluation, ablation and trivial-predictor check.

    venv/bin/python scripts/fairness_utility.py

Reads only stored per-fold predictions:
  results/raw/main/<dbim experiment>/         stage 3 of the ablation, cleaning + resampling (DBIM)
  results/raw/ablation/<impute_only exp.>/    stage 2, cleaning only (configs/ablation.yaml)
  results/raw/baselines/<baseline exp.>/      stage 1 = method "none" (complete rows, no resampling),
                                              plus every Phase 3 baseline
and adds two trivial predictors fitted on the same outer training folds
(sklearn DummyClassifier: "most_frequent" and "stratified", seed 1000 * seed + fold).

Test rows. "complete": the complete rows of each outer test fold, the rows every
method can be scored on (the baselines train on complete rows, author decision B2).
"all": every row of the test fold; only methods with the imputer (DBIM,
cleaning only) and the trivial predictors are scored on them.

Before writing, every metric that a run also stored (fold_metrics.csv) is
recomputed from the predictions and must match it (|diff| < 1e-9).

Writes:
  results/raw/fairness_utility/<experiment>_fold_metrics.csv   per method, test rows, repeat, fold
  results/raw/fairness_utility/<experiment>_group_rates.csv    per method, fold, attribute, group
  results/summary/fairness_utility_summary.csv     mean, SD, corrected 95% CI per method and metric
  results/summary/fairness_utility_<experiment>.csv / .tex     one row per method (mean +- SD)
  results/summary/group_rates_summary.csv          per-group selection rate, TPR, FPR (mean, SD)
  results/summary/ablation_summary.csv / ablation_tests.csv / ablation_<experiment>.tex
  results/figures/tradeoff_<experiment>_<attribute>.png/.pdf
  results/figures/ablation_<experiment>.png/.pdf
  results/summary/fairness_utility_run.json        inputs (sha256), versions, time
"""
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402
from sklearn.dummy import DummyClassifier  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from experiments import stats  # noqa: E402
from experiments.metrics import extended_metrics, group_rate_rows  # noqa: E402
from experiments.protocol import load_experiment_data, outer_splits  # noqa: E402

RAW = REPO_ROOT / "results" / "raw"
OUT_RAW = RAW / "fairness_utility"
OUT_SUM = REPO_ROOT / "results" / "summary"
OUT_FIG = REPO_ROOT / "results" / "figures"

# One analysis per dataset x feature convention (author decision C6).
EXPERIMENTS = [
    {"name": f"{d}_protected_{c}", "dbim": f"main/{d}_dbim_protected_{c}",
     "cleaning_only": f"ablation/{d}_impute_only_protected_{c}",
     "baselines": f"baselines/{d}_baselines_protected_{c}"}
    for d in ("adult", "bank") for c in ("kept", "dropped")
]

UTILITY = ["accuracy", "balanced_accuracy", "f1", "roc_auc", "brier", "ece"]
GAPS = ["dp_diff", "eo_diff", "eopp_diff", "fpr_diff"]
LOWER_IS_BETTER = {"brier", "ece", "dp_diff", "eo_diff", "eopp_diff", "tpr_diff", "fpr_diff"}
STAGES = {"none": "1 original data (complete rows)", "cleaning_only": "2 cleaning only",
          "dbim": "3 cleaning + resampling (DBIM)"}
TRIVIAL = {"trivial_majority": "most_frequent", "trivial_stratified": "stratified"}
NAMES = {"dbim": "DBIM", "cleaning_only": "Cleaning only", "none": "No mitigation",
         "random_over": "Random oversampling", "random_under": "Random undersampling", "smote": "SMOTE",
         "reweighing": "Reweighing", "dir": "DI remover", "exp_gradient": "Exp. gradient",
         "threshold_opt": "Threshold opt.", "trivial_majority": "Majority class",
         "trivial_stratified": "Stratified random"}
MEAN_TOL = 1e-9
INPUTS = []


def run(cmd):
    return subprocess.run(cmd, capture_output=True, text=True).stdout.strip()


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    INPUTS.append(path)
    # round_trip: the default fast parser can be 1 ulp off, which splits tied scores and moves ROC AUC by
    # up to 1.6e-6 (found by the consistency check below).
    return pd.read_csv(path, keep_default_na=False, na_values=[""], low_memory=False, float_precision="round_trip")


def label(method, mitigated):
    return f"{method}[{mitigated}]" if isinstance(mitigated, str) and mitigated else method


def pretty(lab):
    m, _, a = lab.partition("[")
    return NAMES.get(m, m) + (f" [{a}" if a else "")


def experiment_config(dbim_path):
    cfg = yaml.safe_load((REPO_ROOT / "configs" / "main.yaml").read_text())
    exp = next(e for e in cfg["tabular_experiments"] if e["name"] == Path(dbim_path).name)
    return exp, cfg


def check_stored(stored, recomputed, what):
    """Recomputed metrics must equal the metrics the run stored."""
    cols = [c for c in stored.columns if c in recomputed.columns and c not in ("repeat", "fold")]
    s = stored.set_index(["repeat", "fold"])[cols].sort_index()
    r = recomputed.set_index(["repeat", "fold"])[cols].sort_index()
    diff = (s - r).abs().to_numpy()
    worst = np.nanmax(diff)
    if not (worst < MEAN_TOL) or not np.array_equal(np.isnan(s.to_numpy()), np.isnan(r.to_numpy())):
        raise RuntimeError(f"{what}: recomputed metrics differ from stored fold_metrics.csv (max {worst})")
    return worst


def score_sets(pred_sets, groups_cols):
    """pred_sets: list of (label, test_rows, predictions). Returns fold metrics and group rates."""
    metric_rows, group_rows = [], []
    for lab, rows, p in pred_sets:
        for (r, f), q in p.groupby(["repeat", "fold"]):
            g = q[groups_cols].reset_index(drop=True)
            yt, yp, ys = q["y_true"].to_numpy(), q["y_pred"].to_numpy(), q["y_score"].to_numpy()
            metric_rows.append({"method": lab, "test_rows": rows, "repeat": r, "fold": f,
                                "n_test": len(q), **extended_metrics(yt, yp, ys, g)})
            for gr in group_rate_rows(yt, yp, g):
                group_rows.append({"method": lab, "test_rows": rows, "repeat": r, "fold": f, **gr})
    return pd.DataFrame(metric_rows), pd.DataFrame(group_rows)


def trivial_predictions(X, y, groups, strata, cfg, complete):
    """Majority-class and stratified-random predictions on the same outer folds."""
    out = []
    for r, f, seed, tr, te in outer_splits(strata, cfg["outer"]["n_repeats"], cfg["outer"]["n_splits"],
                                           cfg["seeds"]):
        for rows, keep in (("all", np.ones(len(y), bool)), ("complete", complete)):
            tr_k, te_k = tr[keep[tr]], te[keep[te]]
            for method, strategy in TRIVIAL.items():
                clf = DummyClassifier(strategy=strategy, random_state=1000 * seed + f)
                clf.fit(np.zeros((len(tr_k), 1)), y[tr_k])
                Z = np.zeros((len(te_k), 1))
                df = pd.DataFrame({"repeat": r, "fold": f, "seed": seed, "row_id": te_k, "y_true": y[te_k],
                                   "y_pred": clf.predict(Z), "y_score": clf.predict_proba(Z)[:, 1]})
                out.append((method, rows, pd.concat([df, groups.iloc[te_k].reset_index(drop=True)], axis=1)))
    merged = {}
    for method, rows, df in out:
        merged.setdefault((method, rows), []).append(df)
    return [(m, rows, pd.concat(dfs, ignore_index=True)) for (m, rows), dfs in merged.items()]


def analyse(spec):
    exp, cfg = experiment_config(spec["dbim"])
    attrs = list(exp["protected"])
    X, y, groups, strata = load_experiment_data(exp)
    complete = ~X.isna().any(axis=1).to_numpy()
    checks = {}

    dbim = read(RAW / spec["dbim"] / "predictions.csv.gz")
    sets = [("dbim", "all", dbim), ("dbim", "complete", dbim[complete[dbim["row_id"].to_numpy()]])]
    co_dir = RAW / spec["cleaning_only"]
    has_co = (co_dir / "predictions.csv.gz").exists()
    if has_co:
        co = read(co_dir / "predictions.csv.gz")
        sets += [("cleaning_only", "all", co), ("cleaning_only", "complete", co[complete[co["row_id"].to_numpy()]])]
    else:
        print(f"WARNING {spec['name']}: no cleaning-only run yet ({co_dir}); ablation incomplete")
    base = read(RAW / spec["baselines"] / "predictions.csv.gz")
    base["mitigated"] = base["mitigated"].fillna("")
    for (m, a), p in base.groupby(["method", "mitigated"], sort=False):
        sets.append((label(m, a), "complete", p))
    sets += trivial_predictions(X, y, groups, strata, cfg, complete)

    # Every method must be scored on exactly the same test rows per fold.
    ref = {}
    for lab, rows, p in sets:
        for (r, f), q in p.groupby(["repeat", "fold"]):
            key = (rows, r, f)
            ids = frozenset(q["row_id"])
            if ref.setdefault(key, ids) != ids:
                raise RuntimeError(f"{spec['name']} {lab} {rows} r{r} f{f}: test rows differ from other methods")

    fm, gr = score_sets(sets, attrs)

    # Consistency with what the runs stored.
    checks["dbim"] = check_stored(read(RAW / spec["dbim"] / "fold_metrics.csv"),
                                  fm[(fm.method == "dbim") & (fm.test_rows == "all")], f"{spec['name']} dbim")
    if has_co:
        checks["cleaning_only"] = check_stored(read(co_dir / "fold_metrics.csv"),
                                               fm[(fm.method == "cleaning_only") & (fm.test_rows == "all")],
                                               f"{spec['name']} cleaning_only")
    bstored = read(RAW / spec["baselines"] / "fold_metrics.csv")
    bstored["mitigated"] = bstored["mitigated"].fillna("")
    for (m, a), s in bstored.groupby(["method", "mitigated"], sort=False):
        lab = label(m, a)
        checks[lab] = check_stored(s.drop(columns=["method", "mitigated"]), fm[fm.method == lab],
                                   f"{spec['name']} {lab}")
    cc = read(RAW / "comparison" / f"{Path(spec['dbim']).name}_complete_rows_fold_metrics.csv")
    checks["dbim (complete rows, Phase 3 file)"] = check_stored(
        cc.drop(columns=["method", "mitigated"]), fm[(fm.method == "dbim") & (fm.test_rows == "complete")],
        f"{spec['name']} dbim complete rows")

    sel = read(RAW / spec["dbim"] / "selection.csv")
    ratio = float((sel["n_test"] / sel["n_train"]).mean())
    return {"spec": spec, "exp": exp, "attrs": attrs, "fm": fm, "gr": gr, "ratio": ratio, "checks": checks,
            "has_cleaning_only": has_co}


def metric_list(attrs):
    return UTILITY + ["precision", "recall", "positive_rate_pred"] + [f"{a}__{g}" for a in attrs for g in GAPS]


def summarise(res):
    rows = []
    for (lab, tr), g in res["fm"].groupby(["method", "test_rows"], sort=False):
        for met in metric_list(res["attrs"]):
            rows.append({"experiment": res["spec"]["name"], "method": lab, "test_rows": tr, "metric": met,
                         **stats.mean_ci(g[met], res["ratio"])})
    return pd.DataFrame(rows)


def group_summary(res):
    g = res["gr"].groupby(["method", "test_rows", "attribute", "group"], sort=False)
    out = g[["n", "selection_rate", "tpr", "fpr"]].agg(["mean", "std"])
    out.columns = [f"{a}_{b if b == 'mean' else 'sd'}" for a, b in out.columns]
    return out.reset_index().assign(experiment=res["spec"]["name"])


def ablation(res):
    fm = res["fm"]
    tests = []
    pairs = [("cleaning_only", "none"), ("dbim", "cleaning_only"), ("dbim", "none")]
    for rows in ("complete", "all"):
        for met in ["accuracy", "balanced_accuracy", "f1", "roc_auc", "brier", "ece", "positive_rate_pred"] + \
                [f"{a}__{g}" for a in res["attrs"] for g in GAPS]:
            family = []
            for a, b in pairs:
                if rows == "all" and "none" in (a, b):
                    continue  # stage 1 cannot score incomplete rows
                A = fm[(fm.method == a) & (fm.test_rows == rows)].set_index(["repeat", "fold"])[met]
                B = fm[(fm.method == b) & (fm.test_rows == rows)].set_index(["repeat", "fold"])[met]
                if A.empty or B.empty:
                    continue
                d = (A - B).dropna().to_numpy()
                family.append({"experiment": res["spec"]["name"], "test_rows": rows, "metric": met,
                               "direction": "lower is better" if met.split("__")[-1] in LOWER_IS_BETTER
                               else "higher is better",
                               "comparison": f"{STAGES[a]} - {STAGES[b]}", "mean_a": A.mean(), "mean_b": B.mean(),
                               "test_train_ratio": res["ratio"],
                               **stats.corrected_resampled_ttest(d, res["ratio"]),
                               **stats.wilcoxon_signed_rank(d)})
            for r, p1, p2 in zip(family, stats.holm([r["nb_p"] for r in family]),
                                 stats.holm([r["wilcoxon_p"] for r in family])):
                r["nb_p_holm"], r["wilcoxon_p_holm"], r["holm_family_size"] = p1, p2, len(family)
            tests += family
    return pd.DataFrame(tests)


# ---------------------------------------------------------------- tables

def fmt(mean, sd, digits=3):
    return f"{mean:.{digits}f} ± {sd:.{digits}f}"


def wide_table(summary, attrs):
    cols = UTILITY + [f"{a}__{g}" for a in attrs for g in ("dp_diff", "eo_diff", "eopp_diff")]
    s = summary.set_index(["method", "test_rows", "metric"])
    order = summary[["method", "test_rows"]].drop_duplicates()
    rows = []
    for lab, tr in order.itertuples(index=False):
        row = {"method": pretty(lab), "test_rows": tr}
        for c in cols:
            row[c] = fmt(s.loc[(lab, tr, c), "mean"], s.loc[(lab, tr, c), "sd"])
        rows.append(row)
    return pd.DataFrame(rows)


def to_latex(df, caption, label_):
    def esc(x):
        return str(x).replace("_", r"\_").replace("±", r"$\pm$").replace("[", "{[}").replace("]", "{]}")
    head = " & ".join(esc(c) for c in df.columns)
    body = "\n".join(" & ".join(esc(v) for v in r) + r" \\" for r in df.itertuples(index=False))
    return (f"% generated by scripts/fairness_utility.py; mean ± SD over 25 outer folds\n"
            f"\\begin{{table}}[t]\n\\centering\n\\caption{{{caption}}}\n\\label{{{label_}}}\n"
            f"\\resizebox{{\\linewidth}}{{!}}{{%\n\\begin{{tabular}}{{{'l' * 2 + 'r' * (len(df.columns) - 2)}}}\n"
            f"\\toprule\n{head} \\\\\n\\midrule\n{body}\n\\bottomrule\n\\end{{tabular}}}}\n\\end{{table}}\n")


# ---------------------------------------------------------------- figures
# Reference palette (dataviz skill, light mode): three highlighted series, the rest neutral gray.
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
HIGHLIGHT = {"dbim": ("#2a78d6", "o"), "cleaning_only": ("#eb6834", "s"), "none": ("#1baf7a", "D")}
OTHER, TRIVIAL_C = "#8a8984", "#b5b4ae"


def style(ax):
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=8)


def place_labels(ax, pts, gap_px=11, near_px=170):
    """Direct labels right of each point, pushed apart vertically so none overlap; a thin
    leader line joins a label to its point when it had to move."""
    fig = ax.figure
    ax.set_ylim(bottom=min(0, ax.get_ylim()[0]))
    fig.tight_layout(rect=(0, 0.03, 1, 1))  # final layout first, so pixel spacing holds
    fig.canvas.draw()
    to_px, to_data = ax.transData.transform, ax.transData.inverted().transform
    items = sorted(([*to_px((x, y)), text, c] for x, y, text, c in pts), key=lambda t: t[1])
    placed = []  # (x_px, y_px) of labels already placed
    for px, py, text, c in items:
        ly = py
        for qx, qy in placed:  # labels sorted bottom-up: push up past any nearby label below
            if abs(qx - px) < near_px and ly - qy < gap_px:
                ly = qy + gap_px
        placed.append((px, ly))
        moved = abs(ly - py) > 2
        lx, ly_d = to_data((px + 8, ly))
        ax.annotate(text, (to_data((px, py))), xytext=(lx, ly_d), textcoords="data", fontsize=7, color=c,
                    va="center", zorder=4,
                    arrowprops=dict(arrowstyle="-", color=GRID if not moved else "#b5b4ae", lw=0.6,
                                    shrinkA=0, shrinkB=4) if moved else None)


def tradeoff_figure(res, summary, attr):
    """Gap (DP, EO) vs balanced accuracy on complete test rows; mean with corrected 95% CI."""
    s = summary[summary.test_rows == "complete"].set_index(["method", "metric"])
    methods = [m for m in summary[summary.test_rows == "complete"]["method"].unique()
               if ("[" not in m or m.endswith(f"[{attr}]")) and m not in TRIVIAL]  # fairness methods: only attr
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    pts = []
    for ax, gap, title in zip(axes, ("dp_diff", "eo_diff"),
                              ("Demographic-parity difference", "Equalized-odds difference")):
        style(ax)
        # Trivial predictors are not plotted as points (balanced accuracy 0.5 would squeeze the axis):
        # the stratified-random gap is drawn as a band = the gap a model shows by chance alone.
        r = s.loc[("trivial_stratified", f"{attr}__{gap}")]
        ax.axhspan(max(r["ci_low"], 0), r["ci_high"], color=GRID, alpha=0.7, lw=0, zorder=0)
        ax.axhline(r["mean"], color=TRIVIAL_C, lw=1, ls="--", zorder=1)
        for m in methods:
            x, y = s.loc[(m, "balanced_accuracy")], s.loc[(m, f"{attr}__{gap}")]
            base = m.split("[")[0]
            color, marker = HIGHLIGHT.get(base, (TRIVIAL_C if base in TRIVIAL else OTHER,
                                                 "^" if base in TRIVIAL else "o"))
            z = 3 if base in HIGHLIGHT else 2
            ax.errorbar(x["mean"], y["mean"], xerr=[[x["mean"] - x["ci_low"]], [x["ci_high"] - x["mean"]]],
                        yerr=[[y["mean"] - y["ci_low"]], [y["ci_high"] - y["mean"]]], fmt=marker, ms=7,
                        color=color, ecolor=color, elinewidth=1.2, capsize=0, mec="white", mew=1.2, zorder=z)
            pts.append((x["mean"], y["mean"], pretty(m), INK if base in HIGHLIGHT else INK2))
        place_labels(ax, pts)
        pts = []
        ax.set_xlabel("Balanced accuracy (higher is better)", color=INK, fontsize=9)
        ax.set_ylabel(f"{title}, {attr} (lower is better)", color=INK, fontsize=9)
    fig.suptitle(f"{res['spec']['name']}: fairness vs utility for {attr} "
                 f"(complete test rows; mean, corrected 95% CI over 25 outer folds)", fontsize=10, color=INK)
    maj = s.loc[("trivial_majority", "balanced_accuracy"), "mean"]
    fig.text(0.01, 0.01, f"Dashed line and band: gap of a stratified-random predictor (mean, 95% CI), balanced accuracy "
             f"{s.loc[('trivial_stratified', 'balanced_accuracy'), 'mean']:.3f}. Majority-class predictor: "
             f"balanced accuracy {maj:.3f}, all gaps 0 (not shown).", fontsize=7, color=INK2)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    for ext in ("png", "pdf"):
        fig.savefig(OUT_FIG / f"tradeoff_{res['spec']['name']}_{attr}.{ext}", dpi=200)
    plt.close(fig)


def ablation_figure(res, summary):
    if not res["has_cleaning_only"]:
        return
    s = summary[summary.test_rows == "complete"].set_index(["method", "metric"])
    panels = [("balanced_accuracy", "Balanced accuracy"), ("positive_rate_pred", "Predicted positive rate")]
    panels += [(f"{a}__{g}", f"{t}, {a}") for a in res["attrs"] for g, t in (("dp_diff", "DP difference"),
                                                                            ("eo_diff", "EO difference"))]
    fig, axes = plt.subplots(1, len(panels), figsize=(2.6 * len(panels), 3.4))
    order = ["none", "cleaning_only", "dbim"]
    for ax, (met, title) in zip(axes, panels):
        style(ax)
        for i, m in enumerate(order):
            v = s.loc[(m, met)]
            color, marker = HIGHLIGHT[m]
            ax.errorbar(i, v["mean"], yerr=[[v["mean"] - v["ci_low"]], [v["ci_high"] - v["mean"]]], fmt=marker,
                        ms=7, color=color, elinewidth=1.4, capsize=0, mec="white", mew=1.2)
        ax.set_xticks(range(3), ["1 original", "2 cleaning", "3 + resampling"], fontsize=7, rotation=20)
        ax.set_xlim(-0.5, 2.5)
        ax.set_title(title, fontsize=9, color=INK)
    fig.suptitle(f"{res['spec']['name']}: ablation (complete test rows; mean, corrected 95% CI)",
                 fontsize=10, color=INK)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUT_FIG / f"ablation_{res['spec']['name']}.{ext}", dpi=200)
    plt.close(fig)


def main():
    for d in (OUT_RAW, OUT_SUM, OUT_FIG):
        d.mkdir(parents=True, exist_ok=True)
    started = datetime.now(timezone.utc)
    summaries, groups, tests, checks = [], [], [], {}
    for spec in EXPERIMENTS:
        res = analyse(spec)
        name = spec["name"]
        res["fm"].to_csv(OUT_RAW / f"{name}_fold_metrics.csv", index=False)
        res["gr"].to_csv(OUT_RAW / f"{name}_group_rates.csv", index=False)
        summ = summarise(res)
        summaries.append(summ)
        groups.append(group_summary(res))
        abl = ablation(res)
        tests.append(abl)
        checks[name] = {k: float(v) for k, v in res["checks"].items()}

        wt = wide_table(summ, res["attrs"])
        wt.to_csv(OUT_SUM / f"fairness_utility_{name}.csv", index=False)
        (OUT_SUM / f"fairness_utility_{name}.tex").write_text(to_latex(
            wt, f"Utility and fairness on the outer test folds, {name.replace('_', ' ')}.",
            f"tab:fu_{name}"))
        if res["has_cleaning_only"]:
            st = wt[wt["method"].isin([pretty(m) for m in STAGES]) & (wt["test_rows"] == "complete")].copy()
            st["method"] = st["method"].map({pretty(m): v for m, v in STAGES.items()})
            st = st.set_index("method").loc[list(STAGES.values())].reset_index()
            (OUT_SUM / f"ablation_{name}.tex").write_text(to_latex(
                st, f"Ablation, {name.replace('_', ' ')} (complete test rows).", f"tab:ablation_{name}"))
        for a in res["attrs"]:
            tradeoff_figure(res, summ, a)
        ablation_figure(res, summ)
        print(f"{name}: {res['fm']['method'].nunique()} methods, checks max diff "
              f"{max(checks[name].values()):.2e}")

    summ = pd.concat(summaries, ignore_index=True)
    summ.to_csv(OUT_SUM / "fairness_utility_summary.csv", index=False)
    pd.concat(groups, ignore_index=True).to_csv(OUT_SUM / "group_rates_summary.csv", index=False)
    abl = pd.concat(tests, ignore_index=True)
    abl.to_csv(OUT_SUM / "ablation_tests.csv", index=False)
    summ[summ["method"].isin(list(STAGES))].assign(stage=lambda d: d["method"].map(STAGES)).to_csv(
        OUT_SUM / "ablation_summary.csv", index=False)
    record = {"script": "scripts/fairness_utility.py", "started_utc": started.isoformat(timespec="seconds"),
              "finished_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
              "trivial_predictor_seed": "1000 * outer seed + fold",
              "git_commit": run(["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"]),
              "git_dirty": bool(run(["git", "-C", str(REPO_ROOT), "status", "--porcelain", "--untracked-files=no"])),
              "ece_bins": 10, "consistency_checks_max_abs_diff": checks,
              "inputs": {str(Path(p).relative_to(REPO_ROOT)): sha256(p) for p in dict.fromkeys(INPUTS)},
              "packages": {p: metadata.version(p) for p in ["numpy", "pandas", "scipy", "scikit-learn",
                                                            "matplotlib"]}}
    (OUT_SUM / "fairness_utility_run.json").write_text(json.dumps(record, indent=2) + "\n")
    print("done")


if __name__ == "__main__":
    main()
