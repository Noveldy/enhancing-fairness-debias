"""Phase 6: validate the sensitive-attribute detector internally and on external datasets.

    HF_HUB_OFFLINE=1 venv/bin/python scripts/detector_validation.py --config configs/detector.yaml

1. Internal CV on the training descriptions (duplicates removed):
   grouped: StratifiedGroupKFold over paraphrase groups, so templated paraphrases of one
            description never sit on both sides of a split (primary estimate);
   random:  plain StratifiedKFold, the analogue of the original random 80/20 split.
   In both, the epoch is chosen on a validation split carved from the training fold;
   the test fold is used once. The keyword rule applied to the description text is a baseline.
2. Final models (5 seeds) trained on all descriptions; epoch chosen on a grouped validation split.
3. External validation: every column of Adult, Bank Marketing and German Credit (not used in
   the detector's development, author decision D2), target included, with the column
   descriptions from data/loaders.py. Methods: BERT (each seed and majority vote), the keyword
   rule on column names (the app's path without descriptions), and the local checkpoint-830
   (reference only: trained on a random 80% of the same data with the epoch chosen on the test split).
   Ground truth: configs/detector.yaml (author decision L1). Proxies: non-sensitive columns whose
   association with a sensitive column in the data is >= proxy_threshold (Cramer's V,
   correlation ratio or |Spearman|); reported separately, not scored.
4. Overlap: TF-IDF cosine similarity of each external description to its nearest training description.

Writes results/raw/detector/ and results/summary/detector_*.csv, a run record and a log.
"""
import argparse
import json
import logging
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from scipy import stats
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from data import loaders  # noqa: E402
from experiments.detector import (BertDetector, LegacyCheckpoint, keyword_detect, load_training_data,  # noqa: E402
                                  paraphrase_groups)

log = logging.getLogger("detector")
OUT_RAW = REPO_ROOT / "results" / "raw" / "detector"
OUT_SUM = REPO_ROOT / "results" / "summary"


def git(*a):
    return subprocess.run(["git", "-C", str(REPO_ROOT), *a], capture_output=True, text=True).stdout.strip()


def scores(y, p, thr=0.5, prefix=""):
    pred = (np.asarray(p) >= thr).astype(int) if np.asarray(p).dtype.kind == "f" else np.asarray(p)
    out = {"n": len(y), "n_sensitive": int(np.sum(y)), "accuracy": accuracy_score(y, pred),
           "precision": precision_score(y, pred, zero_division=0), "recall": recall_score(y, pred, zero_division=0),
           "f1": f1_score(y, pred, zero_division=0)}
    if np.asarray(p).dtype.kind == "f" and len(set(y)) == 2:
        out["roc_auc"] = roc_auc_score(y, p)
    return {prefix + k: v for k, v in out.items()}


def bert(cfg, seed):
    b = cfg["bert"]
    return BertDetector(seed, lr=b["lr"], batch_size=b["batch_size"], epochs=b["epochs"], threads=b["threads"])


def internal_cv(cfg, d, groups):
    rows, preds, hist = [], [], []
    X, y = d["text"].to_numpy(), d["y"].to_numpy()
    for kind in ("grouped", "random"):
        c = cfg["internal_cv"][kind]
        for r in c["repeats"]:
            if kind == "grouped":
                splits = StratifiedGroupKFold(c["n_splits"], shuffle=True, random_state=r).split(X, y, groups)
            else:
                splits = StratifiedKFold(c["n_splits"], shuffle=True, random_state=r).split(X, y)
            for f, (tr, te) in enumerate(splits):
                if "folds" in c and f not in c["folds"]:     # optional subset (reproduction checks)
                    continue
                nv = cfg["internal_cv"]["validation_fraction_splits"]
                if kind == "grouped":
                    itr, iva = next(StratifiedGroupKFold(nv, shuffle=True, random_state=100 + r).split(
                        X[tr], y[tr], groups[tr]))
                else:
                    itr, iva = next(StratifiedKFold(nv, shuffle=True, random_state=100 + r).split(X[tr], y[tr]))
                a, v = tr[itr], tr[iva]
                m = bert(cfg, seed=1000 * r + f).fit(X[a], y[a], X[v], y[v])
                p = m.predict_proba(X[te])
                kw = np.array([keyword_detect(t) for t in X[te]])
                shared = len(set(groups[te]) & set(groups[a]))
                rows.append({"split": kind, "repeat": r, "fold": f, "n_train": len(a), "n_val": len(v),
                             "n_test": len(te), "best_epoch": m.best_epoch_, "fit_seconds": round(m.fit_seconds_, 1),
                             "test_groups_also_in_train": shared, **scores(y[te], p, cfg["threshold"]),
                             **scores(y[te], kw, prefix="keyword_on_text_")})
                preds.append(pd.DataFrame({"split": kind, "repeat": r, "fold": f, "row": te, "group": groups[te],
                                           "y": y[te], "p_bert": p, "keyword_on_text": kw}))
                hist += [{"split": kind, "repeat": r, "fold": f, **h} for h in m.history_]
                log.info("%s r%d f%d: best epoch %d, acc %.3f, F1 %.3f, AUC %.3f (%.0fs)", kind, r, f, m.best_epoch_,
                         rows[-1]["accuracy"], rows[-1]["f1"], rows[-1]["roc_auc"], m.fit_seconds_)
    return pd.DataFrame(rows), pd.concat(preds), pd.DataFrame(hist)


def association(a, b):
    """Cramer's V (cat-cat), correlation ratio (num-cat) or |Spearman| (num-num)."""
    na, nb = pd.api.types.is_numeric_dtype(a), pd.api.types.is_numeric_dtype(b)
    if na and nb:
        return abs(stats.spearmanr(a, b).statistic)
    if not na and not nb:
        t = pd.crosstab(a, b).to_numpy().astype(float)
        if min(t.shape) < 2:
            return 0.0
        chi2 = stats.chi2_contingency(t, correction=False)[0]
        return float(np.sqrt(chi2 / (t.sum() * (min(t.shape) - 1))))
    cat, x = (a, b) if not na else (b, a)
    x = x.astype(float)
    m = x.groupby(cat.to_numpy()).agg(["mean", "size"])
    ss_b = (m["size"] * (m["mean"] - x.mean()) ** 2).sum()
    return float(np.sqrt(ss_b / ((x - x.mean()) ** 2).sum()))


def external_sets(cfg):
    sets = {"adult": (loaders.load_adult("baseline"), loaders.ADULT_DESCRIPTIONS),
            "bank": (loaders.load_bank("baseline"), loaders.BANK_DESCRIPTIONS),
            "german": (loaders.load_german(), loaders.GERMAN_DESCRIPTIONS)}
    rows = []
    for name, (df, desc) in sets.items():
        sens = cfg["external"][name]["sensitive"]
        assert set(sens) <= set(df.columns) and list(desc) == list(df.columns), name
        for col in df.columns:
            assoc = {s: association(df[col], df[s]) for s in sens if s != col}
            best = max(assoc, key=assoc.get) if assoc else None
            truth = "sensitive" if col in sens else (
                "proxy" if best and assoc[best] >= cfg["proxy_threshold"] else "not sensitive")
            rows.append({"dataset": name, "column": col, "description": desc[col], "truth": truth,
                         "category": sens.get(col, ""), "max_association_with_sensitive": assoc.get(best, np.nan)
                         if best else np.nan, "most_associated_sensitive_column": best or ""})
    return pd.DataFrame(rows)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/detector.yaml")
    ap.add_argument("--out", help="write raw/ and summary/ under this directory instead of results/")
    args = ap.parse_args(argv)
    global OUT_RAW, OUT_SUM
    if args.out:
        OUT_RAW, OUT_SUM = Path(args.out) / "raw" / "detector", Path(args.out) / "summary"
    cfg = yaml.safe_load((REPO_ROOT / args.config).read_text())
    OUT_RAW.mkdir(parents=True, exist_ok=True)
    OUT_SUM.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.StreamHandler(sys.stdout),
                                  logging.FileHandler(OUT_RAW / "run_log.txt", mode="a")])
    started = datetime.now(timezone.utc).isoformat(timespec="seconds")
    t_all = time.perf_counter()
    if git("status", "--porcelain", "--untracked-files=no"):
        log.warning("tracked files have uncommitted changes")

    d, n_dup = load_training_data(REPO_ROOT / cfg["training_data"])
    groups = paraphrase_groups(d["text"], cfg["paraphrase_distance_threshold"])
    d.assign(group=groups).to_csv(OUT_RAW / "training_descriptions_groups.csv", index=False)
    log.info("training descriptions: %d after removing %d duplicates; %d paraphrase groups (largest %d)",
             len(d), n_dup, groups.max() + 1, np.bincount(groups).max())

    # 3 (prepared first: cheap) external ground truth
    ext = external_sets(cfg)
    ext["keyword_on_name"] = [keyword_detect(c) for c in ext["column"]]

    # 1 internal CV
    cv, cv_pred, cv_hist = internal_cv(cfg, d, groups)
    cv.to_csv(OUT_RAW / "internal_cv_folds.csv", index=False)
    cv_pred.to_csv(OUT_RAW / "internal_cv_predictions.csv", index=False)
    cv_hist.to_csv(OUT_RAW / "internal_cv_epochs.csv", index=False)

    # 2 final models and their predictions on the external columns
    X, y = d["text"].to_numpy(), d["y"].to_numpy()
    final = []
    (OUT_RAW / "heads").mkdir(exist_ok=True)
    for seed in cfg["final_models"]["seeds"]:
        a, v = next(StratifiedGroupKFold(cfg["final_models"]["validation_splits"], shuffle=True,
                                         random_state=seed).split(X, y, groups))
        m = bert(cfg, seed).fit(X[a], y[a], X[v], y[v])
        ext[f"p_bert_seed{seed}"] = m.predict_proba(ext["description"])
        torch.save(m.head_state(), OUT_RAW / "heads" / f"bert_head_seed{seed}.pt")
        final.append({"seed": seed, "n_train": len(a), "n_val": len(v), "best_epoch": m.best_epoch_,
                      "fit_seconds": round(m.fit_seconds_, 1), "history": m.history_})
        log.info("final model seed %d: best epoch %d (%.0fs)", seed, m.best_epoch_, m.fit_seconds_)
    seed_cols = [f"p_bert_seed{s}" for s in cfg["final_models"]["seeds"]]
    ext["bert_votes"] = (ext[seed_cols] >= cfg["threshold"]).sum(axis=1)
    ext["bert_majority"] = (ext["bert_votes"] > len(seed_cols) / 2).astype(int)
    legacy_path = REPO_ROOT / cfg["legacy_checkpoint"]
    if legacy_path.exists():
        legacy = LegacyCheckpoint(legacy_path, cfg["bert"]["threads"])
        ext["p_checkpoint830"] = legacy.predict_proba(ext["description"])
    else:   # not in git (4 GB); the reference row is then left out (README, "Detector checkpoint")
        log.warning("%s not found: the checkpoint-830 reference is skipped", legacy_path)
        ext["p_checkpoint830"] = np.nan

    # 4 overlap with the training descriptions
    vec = TfidfVectorizer(stop_words="english").fit(pd.concat([d["text"], ext["description"]]).str.lower())
    sim = cosine_similarity(vec.transform(ext["description"].str.lower()), vec.transform(d["text"].str.lower()))
    ext["nearest_training_similarity"] = sim.max(axis=1)
    ext["nearest_training_description"] = d["text"].to_numpy()[sim.argmax(axis=1)]
    ext["nearest_training_label"] = d["label"].to_numpy()[sim.argmax(axis=1)]
    ext.to_csv(OUT_RAW / "external_predictions.csv", index=False)

    # Summaries
    cvs = []
    for kind, g in cv.groupby("split", sort=False):
        for met in ["accuracy", "precision", "recall", "f1", "roc_auc", "keyword_on_text_accuracy",
                    "keyword_on_text_precision", "keyword_on_text_recall", "keyword_on_text_f1", "best_epoch"]:
            cvs.append({"split": kind, "metric": met, "n_folds": len(g), "mean": g[met].mean(), "sd": g[met].std(),
                        "min": g[met].min(), "max": g[met].max()})
    pd.DataFrame(cvs).to_csv(OUT_SUM / "detector_internal_cv.csv", index=False)

    scored = ext[ext["truth"] != "proxy"]
    ys = (scored["truth"] == "sensitive").astype(int).to_numpy()
    methods = {f"bert seed {s}": (scored[f"p_bert_seed{s}"] >= cfg["threshold"]).astype(int) for s in
               cfg["final_models"]["seeds"]}
    methods["bert majority vote"] = scored["bert_majority"]
    methods["keyword on column name"] = scored["keyword_on_name"]
    if scored["p_checkpoint830"].notna().all():
        methods["checkpoint-830 (reference)"] = (scored["p_checkpoint830"] >= cfg["threshold"]).astype(int)
    rows = []
    for meth, pred in methods.items():
        for ds in ["all"] + list(cfg["external"]):
            mask = np.ones(len(scored), bool) if ds == "all" else (scored["dataset"] == ds).to_numpy()
            r = scores(ys[mask], pred.to_numpy()[mask])
            fp = scored[mask & (pred.to_numpy() == 1) & (ys == 0)]["column"].tolist()
            fn = scored[mask & (pred.to_numpy() == 0) & (ys == 1)]["column"].tolist()
            rows.append({"method": meth, "dataset": ds, **r, "false_positives": ";".join(fp),
                         "false_negatives": ";".join(fn)})
    pd.DataFrame(rows).to_csv(OUT_SUM / "detector_external.csv", index=False)
    cols = ["dataset", "column", "description", "truth", "category", "max_association_with_sensitive",
            "most_associated_sensitive_column", "keyword_on_name", "bert_votes", "bert_majority", "p_checkpoint830",
            "nearest_training_similarity", "nearest_training_description", "nearest_training_label"]
    ext[cols].to_csv(OUT_SUM / "detector_external_columns.csv", index=False)

    record = {"script": "scripts/detector_validation.py", "config": cfg, "started_utc": started,
              "finished_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
              "wall_clock_seconds": round(time.perf_counter() - t_all, 1),
              "training_data_sha256": loaders.sha256(REPO_ROOT / cfg["training_data"]),
              "n_descriptions": len(d), "n_duplicates_removed": n_dup, "n_paraphrase_groups": int(groups.max() + 1),
              "final_models": final,
              "external_data": {k: loaders.RAW_FILES[k] for k in ("adult", "bank", "german")},
              "base_model": {"name": "google-bert/bert-base-uncased", "revision": _hf_revision()},
              "environment": {"python": platform.python_version(), "cpu_threads": cfg["bert"]["threads"],
                              "packages": {p: metadata.version(p) for p in
                                           ["torch", "transformers", "tokenizers", "scikit-learn", "numpy", "pandas"]},
                              "git_commit": git("rev-parse", "HEAD"),
                              "git_dirty": bool(git("status", "--porcelain", "--untracked-files=no"))}}
    (OUT_RAW / "run_detector.json").write_text(json.dumps(record, indent=2, default=str) + "\n")
    log.info("done in %.0f s", record["wall_clock_seconds"])


def _hf_revision():
    ref = Path(os.path.expanduser("~/.cache/huggingface/hub/models--google-bert--bert-base-uncased/refs/main"))
    return ref.read_text().strip() if ref.exists() else "unknown"


if __name__ == "__main__":
    main()
