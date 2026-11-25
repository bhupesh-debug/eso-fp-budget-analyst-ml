"""Multi-year forecasting with honest baselines.

Time-based split: train FY2018-2022, hold out FY2023-2025.
Models: naive (last value), linear trend on year index, Ridge on
[year index, enrollment, prior-year value]. MAPE is the primary metric.
The interpretable baseline is reported alongside the fancier model, and
whichever wins is stated plainly.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler


def _mape(actual: np.ndarray, pred: np.ndarray) -> float:
    return float(np.mean(np.abs((actual - pred) / actual)) * 100)


def _design(years: np.ndarray, enrollment: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Features: year index, enrollment, prior-year value."""
    t = np.arange(len(years), dtype=float)
    prior = np.concatenate([[values[0]], values[:-1]])
    return np.column_stack([t, enrollment, prior])


def forecast_series(years: np.ndarray, values: np.ndarray,
                    enrollment: np.ndarray, n_holdout: int = 3) -> dict:
    years = np.asarray(years, dtype=float)
    values = np.asarray(values, dtype=float)
    enrollment = np.asarray(enrollment, dtype=float)
    n_train = len(years) - n_holdout
    y_tr, y_te = values[:n_train], values[n_train:]
    X = _design(years, enrollment, values)
    X_tr, X_te = X[:n_train], X[n_train:]

    # Naive: last training value carried forward.
    naive_pred = np.full(n_holdout, y_tr[-1])
    # Linear trend on year index only.
    lin = LinearRegression().fit(np.arange(n_train).reshape(-1, 1), y_tr)
    lin_pred = lin.predict(np.arange(n_train, len(years)).reshape(-1, 1))
    # Ridge on full feature set.
    scaler = StandardScaler().fit(X_tr)
    ridge = Ridge(alpha=1.0).fit(scaler.transform(X_tr), y_tr)
    ridge_pred = ridge.predict(scaler.transform(X_te))

    results = {
        "holdout_years": years[n_train:].astype(int).tolist(),
        "actual": y_te.tolist(),
        "naive": {"pred": naive_pred.tolist(), "mape": _mape(y_te, naive_pred)},
        "linear_trend": {"pred": lin_pred.tolist(), "mape": _mape(y_te, lin_pred)},
        "ridge": {"pred": ridge_pred.tolist(), "mape": _mape(y_te, ridge_pred)},
    }
    results["best"] = min(
        ("naive", "linear_trend", "ridge"), key=lambda k: results[k]["mape"])
    return results


def forecast_all(revenue: pd.DataFrame, expenditure: pd.DataFrame,
                 forecast_years: list[int]) -> dict:
    rev_total = revenue.sort_values("fiscal_year")
    exp_yearly = (
        expenditure.groupby("fiscal_year")["actual_expenditure"].sum()
        .reset_index().sort_values("fiscal_year")
    )
    rev = forecast_series(
        rev_total["fiscal_year"].to_numpy(), rev_total["Total Revenue"].to_numpy(),
        rev_total["enrollment"].to_numpy())
    exp = forecast_series(
        exp_yearly["fiscal_year"].to_numpy(), exp_yearly["actual_expenditure"].to_numpy(),
        rev_total["enrollment"].to_numpy())

    # Refit best models on ALL history and project FY2026-28.
    def _project(values, enrollment, years, model_name):
        X = _design(np.asarray(years, dtype=float),
                    np.asarray(enrollment, dtype=float),
                    np.asarray(values, dtype=float))
        if model_name == "naive":
            return [float(values[-1])] * len(forecast_years)
        if model_name == "linear_trend":
            m = LinearRegression().fit(
                np.arange(len(values)).reshape(-1, 1), values)
            return m.predict(
                np.arange(len(values), len(values) + len(forecast_years))
                .reshape(-1, 1)).tolist()
        scaler = StandardScaler().fit(X)
        m = Ridge(alpha=1.0).fit(scaler.transform(X), values)
        # Future features: extend year index, hold enrollment at last value
        # with the historical trend applied, prior = last value rolling.
        preds, last = [], float(values[-1])
        enr = float(enrollment[-1])
        for k in range(len(forecast_years)):
            row = np.array([[len(values) + k, enr, last]])
            p = float(m.predict(scaler.transform(row))[0])
            preds.append(p)
            last = p
            enr *= 0.992  # enrollment trend continues
        return preds

    rev_proj = _project(rev_total["Total Revenue"].to_numpy(),
                        rev_total["enrollment"].to_numpy(),
                        rev_total["fiscal_year"].to_numpy(), rev["best"])
    exp_proj = _project(exp_yearly["actual_expenditure"].to_numpy(),
                        rev_total["enrollment"].to_numpy(),
                        exp_yearly["fiscal_year"].to_numpy(), exp["best"])
    return {
        "revenue": rev, "expenditure": exp,
        "projection": pd.DataFrame({
            "fiscal_year": forecast_years,
            "projected_revenue": [round(v, 2) for v in rev_proj],
            "projected_expenditure": [round(v, 2) for v in exp_proj],
            "projected_surplus_deficit": [
                round(r - e, 2) for r, e in zip(rev_proj, exp_proj)],
        }),
    }
