"""Phase 5 summaries: utility (TSTR vs TRTR), fidelity, privacy, and the TPR/FPR check (5.4).

    venv/bin/python scripts/synthetic_summary.py

Reads results/raw/synthetic/<experiment>/ (configs/synthetic.yaml) and writes:
  results/summary/synthetic_utility.csv          mean, SD, corrected 95% CI per generator, convention, setting
  results/summary/synthetic_utility_tests.csv    TSTR - TRTR per fold (Nadeau-Bengio; Holm across generators)
  results/summary/synthetic_fidelity.csv         joint/marginal fidelity per generator (mean, SD over folds)
  results/summary/synthetic_fidelity_columns.csv per column: KS, TV (mean, SD over folds)
  results/summary/synthetic_fidelity_subgroups.csv per protected group and target class
  results/summary/synthetic_privacy.csv          DCR and membership-inference AUC
  results/summary/synthetic_group_rates.csv      per-group TPR/FPR of TSTR/TRTR models (5.4)
  results/summary/synthetic_<experiment>.tex     utility and fidelity/privacy tables
  results/figures/synthetic_utility_<experiment>.png/.pdf
  results/figures/synthetic_fidelity_columns_<experiment>.png/.pdf
  results/figures/synthetic_tpr_fpr_<experiment>.png/.pdf
  results/summary/synthetic_summary_run.json     inputs (sha256), versions, commit
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

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from experiments import stats  # noqa: E402

RAW = REPO_ROOT / "results" / "raw" / "synthetic"
OUT_SUM = REPO_ROOT / "results" / "summary"
OUT_FIG = REPO_ROOT / "results" / "figures"
EXPERIMENTS = ["adult_synthetic", "bank_synthetic"]
GENERATORS = ["real_train", "marginal_sampler", "independent_marginals", "gaussian_copula"]
NAMES = {"real_train": "Real training data", "marginal_sampler": "Marginal sampler (original FSDG)",
         "independent_marginals": "Independent marginals", "gaussian_copula": "Gaussian copula"}
UTILITY = ["accuracy", "balanced_accuracy", "f1", "roc_auc", "brier", "ece", "positive_rate_pred"]
LOWER = {"brier", "ece", "dp_diff", "eo_diff", "eopp_diff", "tpr_diff", "fpr_diff"}
FID = ["mean_ks_numeric", "max_ks_numeric", "mean_tv", "max_tv", "pairwise_dependence_mae",
       "pairwise_dependence_max", "spearman_diff_frobenius", "c2st_auc"]
PRIV = ["dcr_train_median", "dcr_holdout_median", "exact_copies_train", "exact_copies_holdout",
        "share_closer_to_train", "mia_auc"]
INPUTS = []
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
COLORS = {"real_train": "#1baf7a", "marginal_sampler": "#eb6834", "independent_marginals": "#8a8984",
          "gaussian_copula": "#2a78d6"}
MARKERS = {"real_train": "D", "marginal_sampler": "s", "independent_marginals": "^", "gaussian_copula": "o"}


def sha256(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def read(p):
    INPUTS.append(p)
    return pd.read_csv(p, low_memory=False, float_precision="round_trip")


def git(*args):
    return subprocess.run(["git", "-C", str(REPO_ROOT), *args], capture_output=True, text=True).stdout.strip()


def ratio_for(exp):
    rec = json.loads((RAW / f"run_{exp}.json").read_text())
    INPUTS.append(RAW / f"run_{exp}.json")
    k = rec["config"]["outer"]["n_splits"]
    return 1.0 / (k - 1)          # n_test / n_train of k-fold CV


def utility(exp, ratio):
    t = pd.concat([read(RAW / exp / "tstr_fold_metrics.csv"), read(RAW / exp / "tsts_fold_metrics.csv")])
    attrs = sorted({c.split("__")[0] for c in t.columns if "__" in c})
    mets = UTILITY + [f"{a}__{g}" for a in attrs for g in ("dp_diff", "eo_diff", "eopp_diff", "fpr_diff")]
    rows = []
    for (g, conv, setting), d in t.groupby(["generator", "convention", "setting"], sort=False):
        for m in mets:
            rows.append({"experiment": exp, "generator": g, "convention": conv, "setting": setting, "metric": m,
                         **stats.mean_ci(d[m], ratio)})
    tests = []
    base = t[t.setting == "trtr"]
    for conv in base["convention"].unique():
        b = base[base.convention == conv].set_index(["repeat", "fold"])
        for m in mets:
            fam = []
            for g in GENERATORS[1:]:
                a = t[(t.setting == "tstr") & (t.generator == g) & (t.convention == conv)].set_index(["repeat", "fold"])
                d = (a[m] - b[m]).dropna().to_numpy()
                fam.append({"experiment": exp, "convention": conv, "metric": m,
                            "direction": "lower is better" if m.split("__")[-1] in LOWER else "higher is better",
                            "comparison": f"TSTR {g} - TRTR", "tstr_mean": a[m].mean(), "trtr_mean": b[m].mean(),
                            **stats.corrected_resampled_ttest(d, ratio), **stats.wilcoxon_signed_rank(d)})
            for r, p1, p2 in zip(fam, stats.holm([r["nb_p"] for r in fam]), stats.holm([r["wilcoxon_p"] for r in fam])):
                r["nb_p_holm"], r["wilcoxon_p_holm"] = p1, p2
            tests += fam
    return pd.DataFrame(rows), pd.DataFrame(tests), attrs


def agg(df, by, cols, exp):
    g = df.groupby(by, sort=False)[cols]
    out = g.mean().add_suffix("_mean").join(g.std().add_suffix("_sd")).join(g.size().rename("n_folds"))
    return out.reset_index().assign(experiment=exp)


def fmt(m, s, d=3):
    return f"{m:.{d}f} ± {s:.{d}f}"


def to_latex(df, caption, label):
    def esc(x):
        return str(x).replace("_", r"\_").replace("±", r"$\pm$")
    head = " & ".join(esc(c) for c in df.columns)
    body = "\n".join(" & ".join(esc(v) for v in r) + r" \\" for r in df.itertuples(index=False))
    return (f"\\begin{{table}}[t]\n\\centering\n\\caption{{{caption}}}\n\\label{{{label}}}\n"
            f"\\resizebox{{\\linewidth}}{{!}}{{%\n\\begin{{tabular}}{{l{'r' * (len(df.columns) - 1)}}}\n\\toprule\n"
            f"{head} \\\\\n\\midrule\n{body}\n\\bottomrule\n\\end{{tabular}}}}\n\\end{{table}}\n")


def latex_tables(exp, util, fid, priv, attrs):
    u = util[util.convention == "protected_kept"].set_index(["generator", "setting", "metric"])
    cols = ["roc_auc", "balanced_accuracy", "f1", "brier", "ece"] + [f"{a}__{g}" for a in attrs
                                                                    for g in ("dp_diff", "eo_diff")]
    rows = []
    for g, setting, lab in [("real_train", "trtr", "TRTR (real)")] + \
            [(g, "tstr", f"TSTR {NAMES[g]}") for g in GENERATORS[1:]] + \
            [(g, "tsts", f"TSTS {NAMES[g]}") for g in GENERATORS[1:]]:
        rows.append({"setting": lab, **{c: fmt(u.loc[(g, setting, c), "mean"], u.loc[(g, setting, c), "sd"])
                                        for c in cols}})
    t1 = to_latex(pd.DataFrame(rows), f"Utility of models trained on synthetic data, {exp} (protected attributes "
                  "as inputs; tested on real rows except TSTS; mean ± SD over 25 outer folds).", f"tab:{exp}_utility")
    f = fid.set_index("generator")
    p = priv.set_index("generator")
    rows = []
    for g in GENERATORS:
        rows.append({"generator": NAMES[g],
                     **{c: fmt(f.loc[g, f"{c}_mean"], f.loc[g, f"{c}_sd"]) for c in
                        ["mean_ks_numeric", "mean_tv", "pairwise_dependence_mae", "c2st_auc"]},
                     **{c: fmt(p.loc[g, f"{c}_mean"], p.loc[g, f"{c}_sd"]) for c in
                        ["dcr_train_median", "share_closer_to_train", "mia_auc"]}})
    t2 = to_latex(pd.DataFrame(rows), f"Fidelity against the real test rows and privacy, {exp} (mean ± SD over 25 "
                  "outer folds). 'Real training data' is the sampling-noise floor for fidelity and a copying "
                  "generator for privacy.", f"tab:{exp}_fidelity")
    (OUT_SUM / f"synthetic_{exp}.tex").write_text("% generated by scripts/synthetic_summary.py\n" + t1 + "\n" + t2)


def style(ax):
    ax.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=8)


def utility_figure(exp, util):
    u = util[util.convention == "protected_kept"]
    panels = [("roc_auc", "ROC AUC"), ("balanced_accuracy", "Balanced accuracy"), ("ece", "Calibration error (ECE)")]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
    order = [("real_train", "trtr")] + [(g, "tstr") for g in GENERATORS[1:]] + [(g, "tsts") for g in GENERATORS[1:]]
    labels = ["TRTR"] + [f"TSTR\n{NAMES[g].split(' (')[0]}" for g in GENERATORS[1:]] + \
             [f"TSTS\n{NAMES[g].split(' (')[0]}" for g in GENERATORS[1:]]
    for ax, (m, title) in zip(axes, panels):
        style(ax)
        for i, (g, s) in enumerate(order):
            v = u[(u.generator == g) & (u.setting == s) & (u.metric == m)].iloc[0]
            ax.errorbar(v["mean"], i, xerr=[[v["mean"] - v["ci_low"]], [v["ci_high"] - v["mean"]]],
                        fmt=MARKERS[g], ms=7, color=COLORS[g], mfc=COLORS[g] if s != "tsts" else "white",
                        mec=COLORS[g], elinewidth=1.4, capsize=0)
        if m in ("roc_auc", "balanced_accuracy"):
            ax.axvline(0.5, color=INK2, lw=0.8, ls="--")
        ax.set_yticks(range(len(order)), labels if ax is axes[0] else [""] * len(order), fontsize=7)
        ax.invert_yaxis()
        ax.set_title(title, fontsize=9, color=INK)
    fig.suptitle(f"{exp}: models trained on synthetic data, tested on real rows (TSTS: tested on synthetic rows; "
                 "hollow). Mean, corrected 95% CI", fontsize=9, color=INK)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUT_FIG / f"synthetic_utility_{exp}.{ext}", dpi=200)
    plt.close(fig)


def fidelity_columns_figure(exp, cols):
    c = cols.copy()
    c["dist"] = np.where(c["type"] == "numeric", c["ks_mean"], c["tv_mean"])
    order = c[c.generator == "gaussian_copula"].sort_values("dist")["column"].tolist()
    fig, ax = plt.subplots(figsize=(7, 0.32 * len(order) + 1.2))
    style(ax)
    for g in GENERATORS:
        d = c[c.generator == g].set_index("column").loc[order]
        ax.plot(d["dist"], range(len(order)), MARKERS[g], color=COLORS[g], ms=6, label=NAMES[g], mec="white", mew=0.8)
    ax.set_yticks(range(len(order)), [f"{o} ({'KS' if c[c.column == o]['type'].iloc[0] == 'numeric' else 'TV'})"
                                      for o in order], fontsize=7)
    ax.set_xlabel("Distance to the real test rows (KS for numeric, TV for categorical; mean over folds)",
                  fontsize=8, color=INK)
    ax.legend(fontsize=7, frameon=False, loc="lower right")
    ax.set_title(f"{exp}: per-column marginal fidelity", fontsize=9, color=INK)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUT_FIG / f"synthetic_fidelity_columns_{exp}.{ext}", dpi=200)
    plt.close(fig)


def tpr_fpr_figure(exp, gr):
    d = gr[gr.convention == "protected_kept"]
    attrs = list(d["attribute"].unique())
    fig, axes = plt.subplots(1, len(attrs), figsize=(4.4 * len(attrs), 4.2), squeeze=False)
    for ax, a in zip(axes[0], attrs):
        style(ax)
        ax.plot([0, 1], [0, 1], color=INK2, lw=0.8, ls="--")
        for g in GENERATORS:
            s = d[(d.generator == g) & (d.attribute == a)]
            ax.plot(s["fpr_mean"], s["tpr_mean"], MARKERS[g], color=COLORS[g], ms=7, mec="white", mew=0.8,
                    label=NAMES[g])
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.set_xlabel("False-positive rate", fontsize=8, color=INK)
        ax.set_ylabel("True-positive rate", fontsize=8, color=INK)
        ax.set_title(f"one point per {a} group (mean over folds)", fontsize=8, color=INK)
    axes[0][0].legend(fontsize=7, frameon=False, loc="lower right")
    fig.suptitle(f"{exp}: per-group TPR vs FPR on real test rows. On the diagonal = no class separation",
                 fontsize=9, color=INK)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUT_FIG / f"synthetic_tpr_fpr_{exp}.{ext}", dpi=200)
    plt.close(fig)


def main():
    OUT_SUM.mkdir(parents=True, exist_ok=True)
    OUT_FIG.mkdir(parents=True, exist_ok=True)
    U, T, F, C, S, P, G = [], [], [], [], [], [], []
    for exp in EXPERIMENTS:
        ratio = ratio_for(exp)
        util, tests, attrs = utility(exp, ratio)
        fid = agg(read(RAW / exp / "fidelity.csv"), ["generator"], FID, exp)
        cols = agg(read(RAW / exp / "fidelity_columns.csv"), ["generator", "column", "type"], ["ks", "tv"], exp)
        sub_raw = read(RAW / exp / "fidelity_subgroups.csv")
        sub_raw["subgroup"] = sub_raw["subgroup"].astype(str)
        sub = agg(sub_raw, ["generator", "kind", "subgroup_column", "subgroup"],
                  [c for c in ["n_real", "real_positive_rate", "synth_positive_rate", "mean_ks_numeric", "mean_tv",
                               "pairwise_dependence_mae", "c2st_auc"] if c in sub_raw.columns], exp)
        priv = agg(read(RAW / exp / "privacy.csv"), ["generator"], PRIV, exp)
        gr = agg(read(RAW / exp / "tstr_group_rates.csv"), ["generator", "convention", "attribute", "group"],
                 ["n", "selection_rate", "tpr", "fpr"], exp)
        U.append(util), T.append(tests), F.append(fid), C.append(cols), S.append(sub), P.append(priv), G.append(gr)
        latex_tables(exp, util, fid, priv, attrs)
        utility_figure(exp, util)
        fidelity_columns_figure(exp, cols)
        tpr_fpr_figure(exp, gr)
        print(f"{exp}: done")
    for name, parts in [("synthetic_utility", U), ("synthetic_utility_tests", T), ("synthetic_fidelity", F),
                        ("synthetic_fidelity_columns", C), ("synthetic_fidelity_subgroups", S),
                        ("synthetic_privacy", P), ("synthetic_group_rates", G)]:
        pd.concat(parts, ignore_index=True).to_csv(OUT_SUM / f"{name}.csv", index=False)
    record = {"script": "scripts/synthetic_summary.py",
              "finished_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
              "git_commit": git("rev-parse", "HEAD"),
              "git_dirty": bool(git("status", "--porcelain", "--untracked-files=no")),
              "inputs": {str(Path(p).relative_to(REPO_ROOT)): sha256(p) for p in dict.fromkeys(INPUTS)},
              "packages": {p: metadata.version(p) for p in ["numpy", "pandas", "scipy", "scikit-learn", "matplotlib"]}}
    (OUT_SUM / "synthetic_summary_run.json").write_text(json.dumps(record, indent=2) + "\n")


if __name__ == "__main__":
    main()
