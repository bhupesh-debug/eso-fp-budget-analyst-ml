"""Staffing-cost driver analysis.

Regresses total staffing cost on FTE, average salary and benefit rate
(OLS via LinearRegression) and reports R^2 plus standardised driver
ranking. Also reports per-pupil staffing cost and the salary-vs-benefits
growth split — the two numbers a budget analyst actually briefs.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


def analyze(staffing: pd.DataFrame) -> dict:
    df = staffing.copy()
    X = df[["fte", "avg_salary", "benefit_rate"]].to_numpy(dtype=float)
    y = df["total_staffing_cost"].to_numpy(dtype=float)
    model = LinearRegression().fit(X, y)
    pred = model.predict(X)
    ss_res = float(((y - pred) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1 - ss_res / ss_tot

    # Standardised coefficients for driver ranking.
    X_std = (X - X.mean(axis=0)) / X.std(axis=0)
    y_std = (y - y.mean()) / y.std()
    std_model = LinearRegression().fit(X_std, y_std)
    drivers = sorted(
        zip(["FTE", "Average salary", "Benefit rate"], std_model.coef_),
        key=lambda kv: abs(kv[1]), reverse=True)

    df["per_pupil_staffing_cost"] = (
        df["total_staffing_cost"] / df["enrollment"]).round(0)
    salary_growth = float(
        (df["avg_salary"].iloc[-1] / df["avg_salary"].iloc[0]) ** (1 / (len(df) - 1)) - 1) * 100
    benefit_growth = float(
        (df["benefit_cost"].iloc[-1] / df["benefit_cost"].iloc[0]) ** (1 / (len(df) - 1)) - 1) * 100

    return {
        "r2": round(r2, 3),
        "coefficients": {k: round(float(v), 2) for k, v in
                         zip(["fte", "avg_salary", "benefit_rate"], model.coef_)},
        "intercept": round(float(model.intercept_), 2),
        "driver_ranking": [(k, round(float(v), 3)) for k, v in drivers],
        "per_pupil_staffing_cost_first": float(df["per_pupil_staffing_cost"].iloc[0]),
        "per_pupil_staffing_cost_last": float(df["per_pupil_staffing_cost"].iloc[-1]),
        "avg_salary_cagr_pct": round(salary_growth, 2),
        "benefit_cost_cagr_pct": round(benefit_growth, 2),
        "fte_first": float(df["fte"].iloc[0]),
        "fte_last": float(df["fte"].iloc[-1]),
        "pupils_per_fte_last": float(df["pupils_per_fte"].iloc[-1]),
    }
