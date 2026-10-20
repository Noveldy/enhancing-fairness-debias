"""Phase 5: synthetic data evaluated on the outer folds of the main protocol.

For every outer fold (same seeds and rows as results/raw/main and results/raw/baselines):

1. Each generator is fitted on the complete rows of the outer training fold (all columns:
   features, protected attributes and target) and samples as many rows as it was fitted on.
2. Utility. TSTR: the classifier is trained on the synthetic rows and tested on the real
   complete test rows. TRTR: the same classifier trained on the real training rows (it
   reproduces the Phase 3 'none' baseline; checked). Both use the hyperparameters that the
   Phase 3 'none' inner CV chose in this fold, so synthetic and real models differ only in
   their training data; no tuning on synthetic data. Both feature conventions (protected
   attributes kept / dropped as inputs, author decision C6). TSTS (train on synthetic, test on
   a second, independent synthetic sample) reproduces the setup of the original FSDG reports.
3. Fidelity against the real test rows: per column (KS, TV), joint (association matrix,
   Spearman matrix, classifier two-sample test), per protected group and per target class.
   Row "real_train" compares the real training rows with the real test rows: the level that
   sampling noise alone produces.
4. Privacy: distance to closest record and a distance-based membership-inference attack.
   For "real_train" the "synthetic" data are the training rows themselves (a copying generator).
"""
import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from joblib import Parallel, delayed

from data import loaders
from experiments.memory_guard import GB, MemoryWatchdog, safe_n_jobs
from experiments.metrics import extended_metrics, group_rate_rows
from experiments.pipeline import make_pipeline
from experiments.protocol import load_experiment_data, memory_settings, outer_splits
from experiments.synthetic import fidelity_block, make_generator, privacy_block

log = logging.getLogger(__name__)
REPO_ROOT = Path(__file__).resolve().parents[1]
CONVENTIONS = ("protected_kept", "protected_dropped")


def _hyperparams(exp):
    """(convention, repeat, fold) -> classifier params chosen by the Phase 3 'none' inner CV."""
    out = {}
    for conv, path in exp["hyperparams"].items():
        sel = pd.read_csv(REPO_ROOT / "results" / "raw" / path / "selection.csv")
        for row in sel[sel["method"] == "none"].itertuples():
            # Some stored params are floats (300.0, 5.0): pandas made the column float where max_depth
            # is null. scikit-learn needs ints, so whole-number floats are converted back.
            out[(conv, row.repeat, row.fold)] = {k: int(v) if isinstance(v, float) and v.is_integer() else v
                                                 for k, v in json.loads(row.clf_params).items()}
    return out


def _fit_predict(train, test, target, protected, conv, classifier, seed, params):
    drop = [target] + (list(protected) if conv == "protected_dropped" else [])
    Xtr, Xte = train.drop(columns=drop), test.drop(columns=drop)
    pipe = make_pipeline(Xtr, classifier, seed, impute=False).set_params(
        **{f"clf__{k}": v for k, v in params.items()})
    pipe.fit(Xtr, train[target].to_numpy())
    return pipe.predict(Xte), pipe.predict_proba(Xte)[:, 1]


def _subgroups(real, synth, target, protected, condition_on, min_rows, seed):
    """Fidelity within each protected group and each target class (P(X | group), P(X | Y))."""
    rows = []
    for col in list(protected) + [target]:
        for g in sorted(real[col].unique(), key=str):
            r, s = real[real[col] == g], synth[synth[col] == g]
            row = {"subgroup_column": col, "subgroup": g, "n_real": len(r), "n_synth": len(s),
                   "kind": "target-conditional" if col == target else "protected group"}
            if col != target:
                row["real_positive_rate"] = r[target].mean() if len(r) else np.nan
                row["synth_positive_rate"] = s[target].mean() if len(s) else np.nan
            if len(r) >= min_rows["metrics"] and len(s) >= min_rows["metrics"]:
                # drop the column that defines the subgroup (constant inside it)
                fb, _ = fidelity_block(r.drop(columns=[col]), s.drop(columns=[col]), seed,
                                       c2st=min(len(r), len(s)) >= min_rows["c2st"])
                row.update(fb)
            rows.append(row)
    return rows


def run_fold(exp, cfg, df, r, f, seed, tr, te, hp):
    sklearn.set_config(working_memory=128)   # nearest-neighbour distances in 128 MB blocks
    scfg = cfg["synthetic"]
    target, protected = exp["target"], list(exp["protected"])
    train, test = df.iloc[tr].reset_index(drop=True), df.iloc[te].reset_index(drop=True)
    groups_te = test[protected].reset_index(drop=True)
    out = {k: [] for k in ("tstr", "tsts", "group_rates", "pred", "fid", "fid_cols", "fid_sub", "priv", "timing")}
    key = {"repeat": r, "fold": f, "seed": seed}
    for gi, name in enumerate(["real_train"] + list(scfg["generators"])):
        gseed = 1_000_000 + 10_000 * seed + 100 * f + 2 * gi
        t0 = time.perf_counter()
        if name == "real_train":
            synth = train
        else:
            gen = make_generator(name, target, exp["condition_on"], fit_seed=gseed).fit(train)
            synth = gen.sample(len(train), gseed)
            if name == "marginal_sampler" and r == 0 and f == 0:
                out["schema_example"] = gen.schema_
        t_gen = time.perf_counter() - t0
        t0 = time.perf_counter()
        fb, cf = fidelity_block(test, synth, gseed)
        out["fid"].append({**key, "generator": name, **fb})
        out["fid_cols"] += [{**key, "generator": name, **c} for c in cf.to_dict("records")]
        out["fid_sub"] += [{**key, "generator": name, **s} for s in
                           _subgroups(test, synth, target, protected, exp["condition_on"], scfg["subgroup_min_rows"],
                                      gseed)]
        t_fid = time.perf_counter() - t0
        t0 = time.perf_counter()
        out["priv"].append({**key, "generator": name, **privacy_block(train, test, synth, gseed,
                                                                     scfg["privacy_n_query"])})
        t_priv = time.perf_counter() - t0
        t0 = time.perf_counter()
        synth_test = None if name == "real_train" else gen.sample(len(test), gseed + 1)
        for conv in CONVENTIONS:
            params = hp[(conv, r, f)]
            pred, score = _fit_predict(synth, test, target, protected, conv, exp["classifier"], seed, params)
            label = "trtr" if name == "real_train" else "tstr"
            out["tstr"].append({**key, "generator": name, "convention": conv, "setting": label,
                                **extended_metrics(test[target].to_numpy(), pred, score, groups_te)})
            out["group_rates"] += [{**key, "generator": name, "convention": conv, **g}
                                   for g in group_rate_rows(test[target].to_numpy(), pred, groups_te)]
            out["pred"].append(pd.DataFrame({**key, "generator": name, "convention": conv, "row_id": te, "y_true": test[target].to_numpy(), "y_pred": pred,
                                             "y_score": score}))
            if synth_test is not None:   # the original FSDG setup: train and test on synthetic data
                p2, s2 = _fit_predict(synth, synth_test, target, protected, conv, exp["classifier"], seed, params)
                out["tsts"].append({**key, "generator": name, "convention": conv, "setting": "tsts",
                                    **extended_metrics(synth_test[target].to_numpy(), p2, s2,
                                                       synth_test[protected].reset_index(drop=True))})
        out["timing"].append({**key, "generator": name, "generate_seconds": round(t_gen, 3),
                              "fidelity_seconds": round(t_fid, 2), "privacy_seconds": round(t_priv, 2),
                              "classifier_seconds": round(time.perf_counter() - t0, 2)})
    return out


def run_synthetic_experiment(exp, cfg, space, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    exp = {**exp, "target": loaders.INFO[exp["dataset"]]["target"]}
    if not exp["protected_as_feature"]:
        raise ValueError("generators need the protected attributes as columns: set protected_as_feature: true")
    X, y, groups, strata = load_experiment_data(exp)
    df = X.assign(**{exp["target"]: y})
    complete = ~X.isna().any(axis=1).to_numpy()
    folds = [(r, f, seed, tr[complete[tr]], te[complete[te]])
             for r, f, seed, tr, te in outer_splits(strata, cfg["outer"]["n_repeats"], cfg["outer"]["n_splits"],
                                                    cfg["seeds"], cfg["outer"].get("only"))]
    hp = _hyperparams(exp)
    mem = memory_settings(cfg)
    context = f"{exp['name']} (folds in parallel)"
    n_jobs = safe_n_jobs(cfg["n_jobs"], mem["per_worker_gb"], mem["main_gb"], mem["reserve_gb"], context)
    log.info("%s: %d rows, %d complete, %d folds, generators %s, n_jobs %d", exp["name"], len(df), complete.sum(),
             len(folds), cfg["synthetic"]["generators"], n_jobs)
    t_all = time.perf_counter()
    guard = MemoryWatchdog(mem["budget_gb"], mem["reserve_gb"], context)
    with guard:
        guard.stage = f"generate + evaluate, n_jobs={n_jobs}"
        results = Parallel(n_jobs=n_jobs, verbose=0)(
            delayed(run_fold)(exp, cfg, df, *fold, hp) for fold in folds)
    log.info("%s: %d folds in %.0f s, peak %.1f GB", exp["name"], len(results), time.perf_counter() - t_all,
             guard.peak / GB)

    def cat(k):
        return [x for res in results for x in res[k]]
    pd.DataFrame(cat("tstr")).to_csv(out_dir / "tstr_fold_metrics.csv", index=False)
    pd.DataFrame(cat("tsts")).to_csv(out_dir / "tsts_fold_metrics.csv", index=False)
    pd.DataFrame(cat("group_rates")).to_csv(out_dir / "tstr_group_rates.csv", index=False)
    pd.concat(cat("pred")).to_csv(out_dir / "tstr_predictions.csv.gz", index=False)
    pd.DataFrame(cat("fid")).to_csv(out_dir / "fidelity.csv", index=False)
    pd.DataFrame(cat("fid_cols")).to_csv(out_dir / "fidelity_columns.csv", index=False)
    pd.DataFrame(cat("fid_sub")).to_csv(out_dir / "fidelity_subgroups.csv", index=False)
    pd.DataFrame(cat("priv")).to_csv(out_dir / "privacy.csv", index=False)
    pd.DataFrame(cat("timing")).to_csv(out_dir / "timing.csv", index=False)
    schema = next((res["schema_example"] for res in results if "schema_example" in res), None)
    (out_dir / "marginal_sampler_schema_repeat0_fold0.json").write_text(json.dumps(schema, indent=2) + "\n")
    return {"experiment": exp["name"], "wall_clock_seconds": round(time.perf_counter() - t_all, 1),
            "n_rows": len(df), "n_complete_rows": int(complete.sum()), "n_folds": len(folds),
            "generators": cfg["synthetic"]["generators"], "n_jobs": n_jobs, "memory_guard": mem,
            "peak_memory_gb": round(guard.peak / GB, 2)}
