"""Per-fold utility and fairness metrics on held-out predictions.

``fold_metrics`` is the compact set stored by the Phase 2/3 runs. Phase 4 adds
``extended_metrics`` (expected calibration error, the equal-opportunity gap under
its own name) and ``group_rate_rows`` (per-group selection rate, TPR, FPR), all
recomputed from the stored predictions.

Gap definitions (max - min over the groups of one protected attribute):
  dp_diff    selection rate (demographic parity difference)
  tpr_diff   true-positive rate = equal-opportunity difference (eopp_diff)
  fpr_diff   false-positive rate
  eo_diff    max(tpr_diff, fpr_diff) (equalized-odds difference, as in Fairlearn)
"""
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def _gap(values):
    vals = [v for v in values if not np.isnan(v)]
    return max(vals) - min(vals) if len(vals) >= 2 else np.nan


def group_rates(y_true, y_pred, group):
    """Selection rate, TPR and FPR per group; NaN where a rate is undefined (no positives/negatives)."""
    rates = {}
    for g in sorted(set(group)):
        m = group == g
        yt, yp = y_true[m], y_pred[m]
        pos, neg = yt == 1, yt == 0
        rates[g] = {
            "n": int(m.sum()),
            "selection_rate": float(yp.mean()),
            "tpr": float(yp[pos].mean()) if pos.any() else np.nan,
            "fpr": float(yp[neg].mean()) if neg.any() else np.nan,
        }
    return rates


def fold_metrics(y_true, y_pred, y_score, groups):
    """groups: DataFrame of protected attributes aligned with y_true."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    out = {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_score),
        "brier": brier_score_loss(y_true, y_score),
        "positive_rate_pred": float(y_pred.mean()),
    }
    for attr in groups.columns:
        rates = group_rates(y_true, y_pred, groups[attr].to_numpy())
        out[f"{attr}__dp_diff"] = _gap([r["selection_rate"] for r in rates.values()])
        out[f"{attr}__tpr_diff"] = _gap([r["tpr"] for r in rates.values()])
        out[f"{attr}__fpr_diff"] = _gap([r["fpr"] for r in rates.values()])
        out[f"{attr}__eo_diff"] = np.nanmax([out[f"{attr}__tpr_diff"], out[f"{attr}__fpr_diff"]])
    return out


def expected_calibration_error(y_true, y_score, n_bins=10):
    """Binned ECE: equal-width score bins on [0, 1]; sum over bins of |bin share| * |mean(y) - mean(score)|."""
    y_true = np.asarray(y_true, dtype=float)
    y_score = np.asarray(y_score, dtype=float)
    idx = np.minimum((y_score * n_bins).astype(int), n_bins - 1)  # score 1.0 goes into the last bin
    ece = 0.0
    for b in range(n_bins):
        m = idx == b
        if m.any():
            ece += m.mean() * abs(y_true[m].mean() - y_score[m].mean())
    return float(ece)


def extended_metrics(y_true, y_pred, y_score, groups):
    """fold_metrics plus ECE and the equal-opportunity gap (Phase 4.1)."""
    out = fold_metrics(y_true, y_pred, y_score, groups)
    out["ece"] = expected_calibration_error(y_true, y_score)
    for attr in groups.columns:
        out[f"{attr}__eopp_diff"] = out[f"{attr}__tpr_diff"]
    return out


def group_rate_rows(y_true, y_pred, groups):
    """One row per protected attribute and group: n, selection rate, TPR, FPR."""
    rows = []
    for attr in groups.columns:
        for g, r in group_rates(np.asarray(y_true), np.asarray(y_pred), groups[attr].to_numpy()).items():
            rows.append({"attribute": attr, "group": g, **r})
    return rows
