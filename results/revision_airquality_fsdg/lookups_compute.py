"""Part B lookups: extract stored values, and compute the few that are derived in seconds from stored outputs.

Reads only existing files under results/, configs/ and bias_identification/; reruns nothing. Writes
lookups_computed.json and lookups_computed.md next to this script (LOOKUPS.md cites them).

    docker run --rm -v "$PWD:/repo:ro" \
        -v "$PWD/results/revision_airquality_fsdg:/repo/results/revision_airquality_fsdg" \
        -w /repo debias-revision /app/venv/bin/python results/revision_airquality_fsdg/lookups_compute.py
"""
import hashlib
import json
import sys
from pathlib import Path

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
sys.path[0] = str(ROOT)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

S = ROOT / "results" / "summary"
RAW = ROOT / "results" / "raw"
read = lambda p: pd.read_csv(p, float_precision="round_trip")  # noqa: E731
used, res, md = set(), {}, []


def src(p):
    used.add(str(Path(p).relative_to(ROOT).as_posix()))
    return read(p)


def table(df, title):
    md.append(f"\n## {title}\n")
    md.append(df.to_markdown(index=False, floatfmt=".4g") if hasattr(df, "to_markdown") and _has_tabulate()
              else "```\n" + df.to_string(index=False) + "\n```")


def _has_tabulate():
    try:
        import tabulate  # noqa: F401
        return True
    except ImportError:
        return False


# ---- items 1-2: DBIM vs no mitigation (corrected resampled t, df 24) -----------------------------------------
bt, at = src(S / "baselines_tests.csv"), src(S / "ablation_tests.csv")
keep = ["metric", "mean_diff", "nb_ci_low", "nb_ci_high", "nb_t", "nb_df", "nb_p", "nb_p_holm", "holm_family_size",
        "wilcoxon_p", "wilcoxon_p_holm"]
rows = []
for ds in ("adult", "bank"):
    for conv in ("protected_kept", "protected_dropped"):
        b = bt[(bt.experiment == f"{ds}_baselines_{conv}") & (bt.comparison == "dbim - none")
               & bt.metric.isin(["roc_auc", "accuracy", "balanced_accuracy", "f1"])]
        b = b.assign(source="baselines_tests.csv (dbim - none)", dbim_mean=b.dbim_mean, none_mean=b.baseline_mean)
        a = at[(at.experiment == f"{ds}_{conv}") & (at.test_rows == "complete")
               & at.comparison.str.startswith("3 ") & at.comparison.str.endswith("- 1 original data (complete rows)")]
        # The same comparison in both files: identical differences; t differs only through the test/train
        # ratio used in the Nadeau-Bengio correction (all rows vs complete rows), around the 6th decimal.
        chk = b.merge(a, on="metric", suffixes=("", "_abl"))
        assert np.allclose(chk.mean_diff, chk.mean_diff_abl, rtol=0, atol=1e-15), (ds, conv)
        res.setdefault("t_baselines_vs_ablation_file_max_abs_diff", {})[f"{ds}_{conv}"] = float(
            (chk.nb_t - chk.nb_t_abl).abs().max())
        res.setdefault("test_train_ratio", {})[f"{ds}_{conv}"] = {
            "baselines_tests": float(b.test_train_ratio.iloc[0]), "ablation_tests": float(a.test_train_ratio.iloc[0])}
        e = a[a.metric == "ece"].assign(source="ablation_tests.csv (stage 3 - stage 1, complete test rows)",
                                         dbim_mean=lambda x: x.mean_a, none_mean=lambda x: x.mean_b)
        for part in (b, e):
            for _, r in part.iterrows():
                rows.append({"dataset": ds, "convention": conv, **{k: r[k] for k in keep},
                             "dbim_mean": r.dbim_mean, "none_mean": r.none_mean, "source": r.source})
t79 = pd.DataFrame(rows)
order = {"roc_auc": 0, "accuracy": 1, "balanced_accuracy": 2, "f1": 3, "ece": 4}
t79 = t79.sort_values(["dataset", "convention", "metric"], key=lambda s: s.map(order) if s.name == "metric" else s)
t79.to_csv(OUT / "lookups_dbim_vs_none_tests.csv", index=False)
res["dbim_vs_none"] = t79.to_dict(orient="records")
table(t79[["dataset", "convention", "metric", "dbim_mean", "none_mean", "mean_diff", "nb_ci_low", "nb_ci_high", "nb_t",
           "nb_df", "nb_p", "nb_p_holm", "holm_family_size", "wilcoxon_p", "wilcoxon_p_holm"]],
      "Items 1-2: DBIM vs no mitigation (Nadeau-Bengio corrected t, df 24); ECE from the ablation file")

# ---- item 3: Wilcoxon vs corrected t ---------------------------------------------------------------------------
agree = []
for f, tcol, wcol in (("baselines_tests.csv", "nb_p_holm", "wilcoxon_p_holm"),
                      ("ablation_tests.csv", "nb_p_holm", "wilcoxon_p_holm"),
                      ("synthetic_utility_tests.csv", "nb_p_holm", "wilcoxon_p_holm"),
                      ("timeseries_tests.csv", "t_p_holm", "wilcoxon_p_holm")):
    d = src(S / f)
    ts, ws = d[tcol] < 0.05, d[wcol] < 0.05
    agree.append({"file": f, "t_test": "Nadeau-Bengio corrected t" if tcol == "nb_p_holm"
                  else "plain paired t over 15 runs (p optimistic, see file note)",
                  "comparisons": len(d), "t_significant_holm": int(ts.sum()),
                  "of_which_wilcoxon_also_significant": int((ts & ws).sum()),
                  "t_significant_wilcoxon_not": int((ts & ~ws).sum()), "wilcoxon_only_significant": int((~ts & ws).sum())})
agree = pd.DataFrame(agree)
res["wilcoxon_agreement"] = agree.to_dict(orient="records")
table(agree, "Item 3: significance (Holm < 0.05) of the corrected t-test vs the Wilcoxon signed-rank test")
wonly = t79[(t79.nb_p_holm >= 0.05) & (t79.wilcoxon_p_holm < 0.05)]
res["dbim_vs_none_wilcoxon_only"] = wonly[["dataset", "convention", "metric", "nb_p_holm", "wilcoxon_p_holm"]].to_dict(
    orient="records")
table(wonly[["dataset", "convention", "metric", "mean_diff", "nb_p_holm", "wilcoxon_p_holm"]],
      "Item 3: DBIM vs no mitigation rows significant by Wilcoxon only")

# ---- item 4: synthetic pairwise dependence and real-vs-real KS -------------------------------------------------
fid = src(S / "synthetic_fidelity.csv")
pdep = fid[["experiment", "generator", "pairwise_dependence_mae_mean", "pairwise_dependence_mae_sd",
            "mean_ks_numeric_mean", "mean_ks_numeric_sd"]].rename(
    columns={"mean_ks_numeric_mean": "mean_ks_with_target_mean", "mean_ks_numeric_sd": "mean_ks_with_target_sd"})
ks_rows = []
for exp, target in (("adult_synthetic", "income"), ("bank_synthetic", "y")):
    fc = src(RAW / "synthetic" / exp / "fidelity_columns.csv")
    num = fc[(fc.type == "numeric") & (fc.column != target)]
    per_fold = num.groupby(["generator", "repeat", "fold"]).ks.mean()
    g = per_fold.groupby("generator").agg(["mean", "std", "count"])
    for gen, r in g.iterrows():
        ks_rows.append({"experiment": exp, "generator": gen, "mean_ks_features_only_mean": r["mean"],
                        "mean_ks_features_only_sd": r["std"], "n_folds": int(r["count"]),
                        "n_numeric_features": num.column.nunique()})
ksf = pd.DataFrame(ks_rows)
paper = src(S / "synthetic_paper_means.csv")
chk = ksf.merge(paper, on=["experiment", "generator"])
assert np.allclose(chk.mean_ks_features_only_mean, chk.mean_ks_mean, rtol=0, atol=1e-12)   # same as Phase 10
syn = pdep.merge(ksf, on=["experiment", "generator"])
res["synthetic"] = syn.to_dict(orient="records")
table(syn, "Item 4: pairwise-dependence error and mean KS (with the 0/1 target, and numeric features only; "
           "mean and SD over 25 folds)")

# ---- item 6: resampling triggered / candidates -----------------------------------------------------------------
sel_rows = []
for exp in ("adult_dbim_protected_kept", "adult_dbim_protected_dropped", "bank_dbim_protected_kept",
            "bank_dbim_protected_dropped", "bank_dbim_with_duration"):
    sel = src(RAW / "main" / exp / "selection.csv")
    inner = src(RAW / "main" / exp / "inner_scores.csv.gz")
    names = sorted(inner["sampler"].astype(str).unique())
    per_fold = inner.groupby(["repeat", "fold"]).size()
    sel_rows.append({"experiment": exp, "outer_folds": len(sel),
                     "folds_resampling_triggered": int(sel.resampling_triggered.astype(str).eq("True").sum()),
                     "candidates_per_fold": f"{per_fold.min()}-{per_fold.max()}",
                     "sampler_values_in_inner_scores": ", ".join(names),
                     "none_or_passthrough_candidate": any(n.lower() in ("none", "passthrough") for n in names)})
sel_df = pd.DataFrame(sel_rows)
res["dbim_candidates"] = sel_df.to_dict(orient="records")
table(sel_df, "Item 6: was 'no resampling' a candidate in the DBIM search?")

# ---- item 9: detector -------------------------------------------------------------------------------------------
ext = src(S / "detector_external_columns.csv")
train = src(ROOT / "bias_identification" / "sensitive_features_dataset.csv")
counts = ext.groupby(["dataset", "truth"]).size().unstack(fill_value=0)
counts["scored"] = counts.get("sensitive", 0) + counts.get("not sensitive", 0)
res["detector_columns"] = counts.reset_index().to_dict(orient="records")
table(counts.reset_index(), "Item 9: external columns by dataset and ground truth (proxy columns are not scored)")
texts = set(train.text.str.strip().str.lower())
ext["exact_text_in_training"] = ext.description.str.strip().str.lower().isin(texts)
ov = ext[ext.dataset.isin(["adult", "bank"])][["dataset", "column", "description", "truth", "exact_text_in_training",
                                               "nearest_training_similarity", "nearest_training_description",
                                               "nearest_training_label"]]
res["detector_overlap_adult_bank"] = ov.to_dict(orient="records")
table(ov, "Item 9: Adult/Bank descriptions vs the detector's training texts (exact = identical after lower-casing; "
          "similarity = stored TF-IDF cosine)")
rel = ext[(ext.dataset == "adult") & (ext.column == "relationship")].iloc[0]
res["relationship_proxy"] = {"association": rel.max_association_with_sensitive,
                             "with": rel.most_associated_sensitive_column, "truth": rel.truth}

# ---- item 11: timing ---------------------------------------------------------------------------------------------
cm = src(S / "cost_methods.csv")
none = cm[(cm.dataset == "adult") & (cm.method == "none")].iloc[0]
ft = src(S / "synthetic_fit_time.csv")
real = ft[(ft.experiment == "adult_synthetic") & (ft.training_data == "real_train")]
fsdg = ft[(ft.experiment == "adult_synthetic") & (ft.training_data == "marginal_sampler")]
res["timing_adult_none"] = {
    "cost_methods_model_selection_s": none.model_selection_s, "cost_methods_final_fit_s": none.final_fit_s,
    "cost_methods_classifier_fit_s": none.classifier_fit_s, "cost_methods_total_training_s": none.total_training_s,
    "fit_time_real_train_mean_s": real.fit_seconds.mean(), "fit_time_real_train_folds_s": real.fit_seconds.tolist(),
    "fit_time_fsdg_mean_s": fsdg.fit_seconds.mean(), "fit_time_clf_params": real.clf_params.tolist()}
md.append("\n## Item 11: Adult, no mitigation, timing\n\n```\n" + json.dumps(res["timing_adult_none"], indent=2) + "\n```")

res["inputs_sha256"] = {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in sorted(used)}
(OUT / "lookups_computed.json").write_text(json.dumps(res, indent=2, default=float) + "\n", encoding="utf-8")
(OUT / "lookups_computed.md").write_text(
    "# Part B: values extracted or computed from stored outputs\n\nGenerated by `lookups_compute.py` "
    "(do not edit by hand). Inputs and their sha256: `lookups_computed.json`.\n" + "\n".join(md) + "\n",
    encoding="utf-8")
print("\n".join(md))
