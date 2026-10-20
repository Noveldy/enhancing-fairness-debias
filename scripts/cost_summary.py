"""Phase 7.2-7.3 summary: cost per method and stage, and empirical scaling with the number of rows.

    venv/bin/python scripts/cost_summary.py

Input: results/raw/cost/measurements.csv and tasks.jsonl (scripts/cost_run.py). All times are
single-core wall-clock seconds on the machine in results/hardware.json.

Outputs (results/summary/):
* cost_stages.csv: every timed stage, per dataset and method: n, mean/SD/min/max seconds, CPU seconds,
  peak RSS and RSS increase.
* cost_methods.csv: per dataset and method: model selection (inner CV incl. the final refit), final fit by
  stage, inference, fairness evaluation, total training time, peak memory, and the ratio to no mitigation.
  Fairness baselines reuse the hyperparameters chosen by the 'none' search, so their total is that search
  plus their own fit (their fit and prediction are one call, so inference is included there).
* cost_components.csv: detector, generators, Air Quality cleaning and LSTM.
* cost_scaling.csv: mean time per component and training-set size (timeouts and skips listed);
  cost_scaling_exponents.csv: slope b of log(time) = a + b log(rows) with a 95% CI, local slopes between
  consecutive sizes, and R^2 of the power-law (log-log) and exponential (log-linear) fits.
Figures (results/figures/): cost_methods_<dataset>, cost_scaling_<dataset> (.png/.pdf).
"""
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
import matplotlib.ticker
import numpy as np
import pandas as pd
from scipy import stats as st

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
RAW = REPO_ROOT / "results" / "raw" / "cost"
OUT_SUM, OUT_FIG = REPO_ROOT / "results" / "summary", REPO_ROOT / "results" / "figures"
INK, INK2, GRID, GRAY = "#0b0b0b", "#52514e", "#e4e3df", "#a3a29d"
BLUE, ORANGE, GREEN = "#2a78d6", "#eb6834", "#1baf7a"
METHODS = ["dbim", "none", "random_over", "random_under", "smote", "reweighing", "dir", "exp_gradient",
           "threshold_opt"]
NAMES = {"dbim": "DBIM (proposed)", "none": "No mitigation", "random_over": "Random oversampling",
         "random_under": "Random undersampling", "smote": "SMOTE", "reweighing": "Reweighing",
         "dir": "Disparate impact remover", "exp_gradient": "Exponentiated gradient",
         "threshold_opt": "Threshold optimizer"}
FAIR = {"reweighing", "dir", "exp_gradient", "threshold_opt"}
FIT_STAGES = ["imbalance_check", "cleaning", "encoding", "resampling", "classifier_fit", "fit_and_inference"]


def sha256(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def git(*args):
    return subprocess.run(["git", "-C", str(REPO_ROOT), *args], capture_output=True, text=True).stdout.strip()


def agg(g):
    s = g["seconds"]
    return pd.Series({"n": len(s), "mean_s": s.mean(), "sd_s": s.std(ddof=1) if len(s) > 1 else np.nan,
                      "min_s": s.min(), "max_s": s.max(), "mean_cpu_s": g["cpu_seconds"].mean(),
                      "max_peak_rss_gb": g["peak_rss_gb"].max(), "mean_rss_increase_gb": g["rss_increase_gb"].mean()})


# ----------------------------------------------------------------------------- 7.2
def stage_table(m):
    c = m[m["part"] == "cost"].copy()
    keys = ["dataset", "task", "method", "stage"] + (["scenario"] if "scenario" in c else [])
    return c.groupby(keys, dropna=False).apply(agg, include_groups=False).reset_index()


def method_table(m):
    c = m[(m["part"] == "cost") & m["task"].isin(["dbim_fit", "baseline_fit", "search"])]
    rows = []
    for ds in sorted(c["dataset"].unique()):
        d = c[c["dataset"] == ds]
        search = d[d["task"] == "search"].groupby("method")["seconds"]
        search_peak = d[d["task"] == "search"].groupby("method")["peak_rss_gb"].max()
        for meth in METHODS:
            fit = d[(d["task"] != "search") & (d["method"] == meth)]
            if fit.empty:
                continue
            per_stage = fit.groupby("stage")["seconds"].mean()
            per_fold_fit = fit[fit["stage"].isin(FIT_STAGES)].groupby("fold")["seconds"].sum()
            sel_key = "none" if meth in FAIR else meth
            sel = search.mean().get(sel_key, np.nan)
            r = {"dataset": ds, "method": meth, "name": NAMES[meth], "n_folds": fit["fold"].nunique(),
                 "n_train_rows": fit["n_train_rows"].mean(),
                 "model_selection_s": sel, "model_selection_sd_s": search.std().get(sel_key, np.nan),
                 "model_selection_source": f"inner CV of '{sel_key}'",
                 "final_fit_s": per_fold_fit.mean(), "final_fit_sd_s": per_fold_fit.std(ddof=1),
                 **{f"{s}_s": per_stage.get(s, np.nan) for s in FIT_STAGES},
                 "inference_s": per_stage.get("inference", np.nan),
                 "fairness_evaluation_s": per_stage.get("fairness_evaluation", np.nan),
                 "total_training_s": sel + (per_fold_fit.mean() if meth in FAIR else 0.0),
                 "peak_rss_gb": max(fit["peak_rss_gb"].max(), search_peak.get(sel_key, 0.0)),
                 "inference_included_in_fit": meth in FAIR}
            rows.append(r)
    out = pd.DataFrame(rows)
    base = out[out["method"] == "none"].set_index("dataset")["total_training_s"]
    out["total_vs_none"] = out["total_training_s"] / out["dataset"].map(base)
    return out


def component_table(m):
    c = m[(m["part"] == "cost") & m["task"].isin(["detection", "generator", "ts_cleaning", "ts_lstm"])].copy()
    keys = ["task", "dataset", "method", "stage"]
    out = c.groupby(keys, dropna=False).apply(agg, include_groups=False).reset_index()
    if "epochs" in c:
        ep = c[c["stage"] == "forecaster_fit"].groupby(["method", "scenario"])["epochs"].first()
        out.attrs["lstm_epochs"] = ep.to_dict()
    return out


# ----------------------------------------------------------------------------- 7.3
def scaling_points(m, tasks):
    """One row per (dataset, component, size point) measurement."""
    pts = []
    sc = m[(m["dataset"].isin(["adult", "bank"])) & (m["task"].isin(["dbim_fit", "resampler", "generator", "search"]))]
    for _, r in sc.iterrows():
        frac, part = r["frac"], r["part"]
        if r["task"] == "dbim_fit":
            if r["stage"] == "cleaning":
                comp = "cleaning (similarity imputer)"
            elif r["stage"] == "classifier_fit" and (frac < 1 or r["fold"] == 0):
                comp = "classifier fit (fixed configuration)"   # 100%: fold 0 has the same configuration
            elif r["stage"] == "inference" and (frac < 1 or r["fold"] == 0):
                comp = "inference (whole test fold)"
            else:
                continue
        elif r["task"] == "resampler":
            if r.get("status", "ok") != "ok":
                continue
            comp = f"resampler: {r['resampler']}"
        elif r["task"] == "generator":
            if part == "cost" and r["method"] not in ("marginal_sampler", "gaussian_copula"):
                continue
            comp = f"generator: {r['method']} ({'fit' if r['stage'] == 'generation_fit' else 'sample'})"
        elif r["task"] == "search":
            if r["method"] != "dbim":
                continue
            comp = "DBIM model selection (inner CV)"
        pts.append({"dataset": r["dataset"], "component": comp, "frac": frac, "fold": r["fold"],
                    "sub_seed": r["sub_seed"], "n_train_rows": r["n_train_rows"], "seconds": r["seconds"],
                    "peak_rss_gb": r["peak_rss_gb"], "n_resamplers": r.get("n_resamplers", np.nan)})
    pts = pd.DataFrame(pts)
    # timeouts / skips / errors of scaling tasks
    bad = []
    for t in tasks.values():
        s = t["spec"]
        if t["status"] != "ok" and s["task"] in ("resampler", "search") and s.get("dataset") in ("adult", "bank"):
            comp = f"resampler: {s['resampler']}" if s["task"] == "resampler" else "DBIM model selection (inner CV)"
            bad.append({"dataset": s["dataset"], "component": comp, "frac": s.get("frac"), "fold": s.get("fold"),
                        "sub_seed": s.get("sub_seed"), "status": t["status"], "message": t["message"][:200],
                        "seconds_before_stop": t["seconds_total"]})
    for t in tasks.values():   # resampler calls that raised inside an ok task
        for r in t["rows"]:
            if r.get("status") == "error":
                bad.append({"dataset": r["dataset"], "component": f"resampler: {r['resampler']}", "frac": r["frac"],
                            "fold": r["fold"], "sub_seed": r["sub_seed"], "status": "error",
                            "message": r.get("error", "")[:200], "seconds_before_stop": np.nan})
    return pts, pd.DataFrame(bad)


def scaling_tables(pts, bad):
    tab = (pts.groupby(["dataset", "component", "frac"])
           .agg(n=("seconds", "size"), n_train_rows=("n_train_rows", "mean"), mean_s=("seconds", "mean"),
                sd_s=("seconds", "std"), max_peak_rss_gb=("peak_rss_gb", "max"),
                n_resamplers=("n_resamplers", "max")).reset_index())
    if not bad.empty:
        b = bad.groupby(["dataset", "component", "frac", "status"]).size().unstack("status", fill_value=0)
        tab = tab.merge(b.reset_index(), on=["dataset", "component", "frac"], how="outer")
    exps = []
    for (ds, comp), g in pts.groupby(["dataset", "component"]):
        g = g[g["seconds"] > 0]
        if g["frac"].nunique() < 3:
            continue
        x, y = np.log(g["n_train_rows"].to_numpy(float)), np.log(g["seconds"].to_numpy(float))
        lr = st.linregress(x, y)
        h = st.t.ppf(0.975, len(x) - 2) * lr.stderr
        semilog = st.linregress(g["n_train_rows"].to_numpy(float), y)
        means = g.groupby("frac").agg(n=("n_train_rows", "mean"), t=("seconds", "mean")).sort_index()
        local = np.diff(np.log(means["t"])) / np.diff(np.log(means["n"]))
        exps.append({"dataset": ds, "component": comp, "n_points": len(x), "sizes": means.index.tolist(),
                     "rows_min": means["n"].min(), "rows_max": means["n"].max(),
                     "seconds_at_min": means["t"].iloc[0], "seconds_at_max": means["t"].iloc[-1],
                     "exponent_b": lr.slope, "exponent_ci_low": lr.slope - h, "exponent_ci_high": lr.slope + h,
                     "r2_power_law": lr.rvalue ** 2, "r2_exponential": semilog.rvalue ** 2,
                     "local_exponents": json.dumps([round(v, 2) for v in local]),
                     "note": "search at 10-50% includes SVMSMOTE and CNN (< 20,000 training rows), at 100% not"
                     if comp.startswith("DBIM") else ""})
    return tab, pd.DataFrame(exps)


# ----------------------------------------------------------------------------- figures
def style(ax):
    ax.grid(True, color=GRID, lw=0.8, which="major")
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=8)


def fmt_s(s):
    return f"{s:.2g} s" if s < 10 else (f"{s:.0f} s" if s < 120 else f"{s / 60:.1f} min")


def methods_figure(ds, mt):
    d = mt[mt["dataset"] == ds].set_index("method").reindex(METHODS).dropna(subset=["name"])
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), gridspec_kw={"width_ratios": [1.6, 1]})
    y = np.arange(len(d))
    for ax, col, title, log in [(axes[0], "total_training_s", "Training time, incl. model selection "
                                 "(single core, log scale)", True),
                                (axes[1], "peak_rss_gb", "Peak memory of the process (GB, RSS)", False)]:
        style(ax)
        colors = [BLUE if m == "dbim" else GRAY for m in d.index]
        if log:   # dots, not bars: a bar's length on a log axis depends on an arbitrary baseline
            ax.scatter(d[col], y, c=colors, s=40, zorder=3)
            ax.set_xscale("log")
            ax.set_xlim(d[col].min() / 2, d[col].max() * 6)
        else:
            ax.barh(y, d[col], color=colors, height=0.6)
        for yi, v in zip(y, d[col]):
            ax.text(v * 1.25 if log else v + 0.02, yi, fmt_s(v) if log else f"{v:.2f}",
                    va="center", fontsize=7, color=INK)
        ax.set_yticks(y, d["name"] if ax is axes[0] else [""] * len(d), fontsize=8)
        ax.invert_yaxis()
        ax.set_title(title, fontsize=9, color=INK)
    axes[1].set_xlim(0, d["peak_rss_gb"].max() * 1.3)
    fig.suptitle(f"{ds}: cost per method, mean of 3 outer folds (fairness methods: 'none' search + own fit)",
                 fontsize=9, color=INK)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUT_FIG / f"cost_methods_{ds}.{ext}", dpi=200)
    plt.close(fig)


def spread_labels(ax, items, gap=0.17, **kw):
    """Direct labels at the right end of each line, nudged apart vertically (log y) so none overlap."""
    items = sorted(items, key=lambda i: i[1])
    ys = [np.log10(y) for _, y, _, _ in items]
    for i in range(1, len(ys)):
        ys[i] = max(ys[i], ys[i - 1] + gap)
    for (x, _, text, color), y in zip(items, ys):
        ax.text(x * 1.08, 10 ** y, text, color=color, va="center", **kw)


def scaling_figure(ds, tab, bad, threshold):
    t = tab[(tab["dataset"] == ds) & tab["mean_s"].notna()]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    stages = {"DBIM model selection (inner CV)": ORANGE, "cleaning (similarity imputer)": BLUE,
              "classifier fit (fixed configuration)": GREEN, "generator: gaussian_copula (fit)": GRAY,
              "generator: marginal_sampler (fit)": GRAY}
    labels = {"DBIM model selection (inner CV)": "DBIM model selection", "cleaning (similarity imputer)":
              "Cleaning (imputer)", "classifier fit (fixed configuration)": "Classifier fit",
              "generator: gaussian_copula (fit)": "Copula fit", "generator: marginal_sampler (fit)":
              "Marginal sampler fit"}
    ax = axes[0]
    style(ax)
    lab = []
    for comp, col in stages.items():
        g = t[t["component"] == comp].sort_values("n_train_rows")
        if g.empty:
            continue
        ax.plot(g["n_train_rows"], g["mean_s"], "-o", color=col, lw=2, ms=4)
        lab.append((g["n_train_rows"].iloc[-1], g["mean_s"].iloc[-1], labels[comp], INK))
    spread_labels(ax, lab, fontsize=7)
    ax.set_title("Pipeline stages", fontsize=9, color=INK)
    ax = axes[1]
    style(ax)
    hi = {"resampler: CondensedNearestNeighbour": ORANGE, "resampler: SVMSMOTE": GREEN}
    lab = []
    for comp, g in t[t["component"].str.startswith("resampler")].groupby("component"):
        g = g.sort_values("n_train_rows")
        col = hi.get(comp, GRAY)
        ax.plot(g["n_train_rows"], g["mean_s"], "-o", color=col, lw=2 if comp in hi else 1.2, ms=3,
                zorder=3 if comp in hi else 2)
        lab.append((g["n_train_rows"].iloc[-1], g["mean_s"].iloc[-1], comp.split(": ")[1],
                    INK if comp in hi else INK2))
    spread_labels(ax, lab, fontsize=6.5)
    b = bad[(bad["dataset"] == ds) & (bad["status"] == "timeout")] if not bad.empty else bad
    if not b.empty:
        for _, r in b.iterrows():
            rows = t[(t["frac"] == r["frac"])]["n_train_rows"].mean()
            ax.plot(rows, r["seconds_before_stop"], "x", color=INK, ms=7)
    ax.set_title("Resamplers" + (" (x = stopped at the time limit)" if not b.empty else ""), fontsize=9,
                 color=INK)
    for a in axes:
        a.axvline(threshold, color=INK2, lw=0.8, ls="--")
        a.text(threshold * 0.95, a.get_ylim()[0] * 1.5, f"search drops SVMSMOTE and CNN\nabove {threshold:,} rows",
               fontsize=6.5, color=INK2, ha="right", va="bottom")
        a.set_xscale("log")
        a.set_yscale("log")
        a.set_xlabel("training rows", fontsize=8, color=INK2)
        a.set_xlim(right=a.get_xlim()[1] * 2.2)
        a.xaxis.set_major_locator(matplotlib.ticker.FixedLocator([2000, 5000, 10000, 20000, 50000]))
        a.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        a.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
        a.tick_params(axis="x", labelsize=7)
    axes[0].set_ylabel("seconds (single core)", fontsize=8, color=INK2)
    fig.suptitle(f"{ds}: time vs. training-set size (10/25/50/100% of an outer training fold; mean of 3)",
                 fontsize=9, color=INK)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUT_FIG / f"cost_scaling_{ds}.{ext}", dpi=200)
    plt.close(fig)


def main(argv=None):
    import argparse
    import yaml
    global RAW, OUT_SUM, OUT_FIG
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", default=str(RAW))
    ap.add_argument("--out-sum", default=str(OUT_SUM))
    ap.add_argument("--out-fig", default=str(OUT_FIG))
    a = ap.parse_args(argv)
    RAW, OUT_SUM, OUT_FIG = Path(a.raw), Path(a.out_sum), Path(a.out_fig)
    inputs = [RAW / "measurements.csv", RAW / "tasks.jsonl", RAW / "run_cost.json"]
    rel = [p.relative_to(REPO_ROOT) if p.is_relative_to(REPO_ROOT) else p for p in inputs]
    m = pd.read_csv(inputs[0], float_precision="round_trip", low_memory=False)
    tasks = {}
    for line in inputs[1].read_text().splitlines():
        rec = json.loads(line)
        tasks[rec["task_id"]] = rec          # the last record of a task wins (resumed runs)
    stage_table(m).to_csv(OUT_SUM / "cost_stages.csv", index=False)
    mt = method_table(m)
    mt.to_csv(OUT_SUM / "cost_methods.csv", index=False)
    component_table(m).to_csv(OUT_SUM / "cost_components.csv", index=False)
    pts, bad = scaling_points(m, tasks)
    tab, exps = scaling_tables(pts, bad)
    tab.to_csv(OUT_SUM / "cost_scaling.csv", index=False)
    exps.to_csv(OUT_SUM / "cost_scaling_exponents.csv", index=False)
    if not bad.empty:
        bad.to_csv(OUT_SUM / "cost_scaling_failures.csv", index=False)
    space = yaml.safe_load((REPO_ROOT / "configs" / "search_space.yaml").read_text())
    for ds in sorted(mt["dataset"].unique()):
        methods_figure(ds, mt)
        scaling_figure(ds, tab, bad, space["resamplers"]["large_training_rows"])
    record = {"script": "scripts/cost_summary.py", "created_utc": datetime.now(timezone.utc).isoformat(
        timespec="seconds"), "git_commit": git("rev-parse", "HEAD"),
        "git_dirty": bool(git("status", "--porcelain", "--untracked-files=no")),
        "inputs": {str(r): sha256(p) for r, p in zip(rel, inputs)},
        "task_status": pd.Series([t["status"] for t in tasks.values()]).value_counts().to_dict()}
    (OUT_SUM / "cost_summary_run.json").write_text(json.dumps(record, indent=2) + "\n")
    print(record["task_status"])


if __name__ == "__main__":
    sys.exit(main())
