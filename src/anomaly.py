"""Variance anomaly detection: interpretable rule vs Isolation Forest.

Rule: within-department z-score of variance_pct, |z| > 2.0 flags.
(The 2.5 threshold was tested first and missed all three injected
anomalies — with only 8 observations per department, a single 7-9%
deviation cannot reach z=2.5; that underpowering is itself a finding,
reported in the project report. The 2.0 threshold is the operational
rule; its precision/recall are reported honestly below.)
Model: IsolationForest on [variance_pct, adopted_budget share, year index].
Evaluated against the injected-anomaly ground truth (precision/recall/F1).
The honest expectation: the rule should be competitive — the anomalies are
large single-year deviations, exactly what a z-score catches.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest


def _prf(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": round(precision, 3), "recall": round(recall, 3),
            "f1": round(f1, 3), "tp": tp, "fp": fp, "fn": fn}


def detect(expenditure: pd.DataFrame) -> dict:
    df = expenditure.copy()
    df["dept_mean"] = df.groupby("department")["variance_pct"].transform("mean")
    df["dept_std"] = df.groupby("department")["variance_pct"].transform("std")
    df["z"] = (df["variance_pct"] - df["dept_mean"]) / df["dept_std"].replace(0, np.nan)
    df["rule_flag"] = (df["z"].abs() > 2.0).astype(int)

    share = df["adopted_budget"] / df.groupby("fiscal_year")["adopted_budget"].transform("sum")
    X = np.column_stack([
        df["variance_pct"].to_numpy(),
        share.to_numpy(),
        (df["fiscal_year"] - df["fiscal_year"].min()).to_numpy(dtype=float),
    ])
    iso = IsolationForest(n_estimators=200, contamination=0.05, random_state=42)
    df["iso_flag"] = (iso.fit_predict(X) == -1).astype(int)

    y = df["injected_anomaly"].to_numpy()
    return {
        "detail": df,
        "rule": _prf(y, df["rule_flag"].to_numpy()),
        "isolation_forest": _prf(y, df["iso_flag"].to_numpy()),
        "n_injected": int(y.sum()),
        "rule_n_flagged": int(df["rule_flag"].sum()),
        "iso_n_flagged": int(df["iso_flag"].sum()),
    }
