"""Budget-vs-actual variance analysis."""
from __future__ import annotations

import numpy as np
import pandas as pd

FLAG_THRESHOLD = 0.05  # |variance| > 5% gets flagged for review


def analyze(expenditure: pd.DataFrame) -> dict:
    df = expenditure.copy()
    df["abs_variance_pct"] = df["variance_pct"].abs()
    df["flagged"] = df["abs_variance_pct"] > FLAG_THRESHOLD

    by_year = (
        df.groupby("fiscal_year")
        .agg(total_adopted=("adopted_budget", "sum"),
             total_actual=("actual_expenditure", "sum"))
        .reset_index()
    )
    by_year["variance_pct"] = (
        (by_year["total_actual"] - by_year["total_adopted"]) / by_year["total_adopted"]
    ).round(4)

    by_dept = (
        df.groupby("department")
        .agg(mean_abs_variance_pct=("abs_variance_pct", "mean"),
             max_abs_variance_pct=("abs_variance_pct", "max"),
             n_flagged=("flagged", "sum"),
             n_years=("fiscal_year", "count"))
        .reset_index()
        .sort_values("mean_abs_variance_pct", ascending=False)
    )

    return {
        "detail": df,
        "by_year": by_year,
        "by_dept": by_dept,
        "mean_abs_variance_pct": float(df["abs_variance_pct"].mean()),
        "n_flagged": int(df["flagged"].sum()),
        "worst_rows": df.sort_values("abs_variance_pct", ascending=False).head(5),
    }
