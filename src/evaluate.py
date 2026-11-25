"""Aggregate every module's outputs into reports/metrics.json."""
from __future__ import annotations

import json

import numpy as np


def _f(x):
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, np.ndarray):
        return x.tolist()
    return x


def build(var, forecasts, score, projected_fund, staff, scenarios_df, anom) -> dict:
    scen = scenarios_df.set_index("scenario").to_dict("index")
    return {
        "synthetic_data": True,
        "history_years": [2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025],
        "forecast_years": [2026, 2027, 2028],
        "variance": {
            "mean_abs_variance_pct": round(var["mean_abs_variance_pct"] * 100, 2),
            "n_flagged_over_5pct": var["n_flagged"],
            "worst_department": var["by_dept"].iloc[0]["department"],
            "worst_department_mean_abs_variance_pct": round(
                float(var["by_dept"].iloc[0]["mean_abs_variance_pct"]) * 100, 2),
        },
        "forecast_holdout_mape_pct": {
            "revenue": {k: round(v["mape"], 2) for k, v in
                        ((k, forecasts["revenue"][k]) for k in
                         ("naive", "linear_trend", "ridge"))},
            "expenditure": {k: round(v["mape"], 2) for k, v in
                            ((k, forecasts["expenditure"][k]) for k in
                             ("naive", "linear_trend", "ridge"))},
            "best_revenue_model": forecasts["revenue"]["best"],
            "best_expenditure_model": forecasts["expenditure"]["best"],
        },
        "projection_fy26_28": forecasts["projection"].to_dict("records"),
        "sustainability": score,
        "projected_fund_balance": projected_fund.to_dict("records"),
        "staffing": staff,
        "scenarios": {k: {kk: _f(vv) for kk, vv in v.items()} for k, v in scen.items()},
        "anomaly_detection": {
            "rule_zscore": anom["rule"],
            "isolation_forest": anom["isolation_forest"],
            "n_injected_anomalies": anom["n_injected"],
        },
    }


def write(metrics: dict, path: str) -> None:
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2, default=_f)
