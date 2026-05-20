from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def safe_auc(labels: list[int], scores: list[float]) -> dict[str, float | None]:
    if len(set(labels)) < 2:
        return {"roc_auc": None, "pr_auc": None}
    return {
        "roc_auc": float(roc_auc_score(labels, scores)),
        "pr_auc": float(average_precision_score(labels, scores)),
    }


def spearman(x: list[float], y: list[float]) -> float | None:
    if len(x) < 2 or len(y) < 2:
        return None
    xr = np.argsort(np.argsort(np.asarray(x)))
    yr = np.argsort(np.argsort(np.asarray(y)))
    return float(np.corrcoef(xr, yr)[0, 1])
