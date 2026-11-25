"""End-to-end pipeline: generate synthetic data -> analyse -> forecast ->
fund balance -> staffing -> scenarios -> anomalies -> charts -> metrics.

Usage: python run_all.py
All data is SYNTHETIC. Nothing here is the real finances of Phoenix Central
School District or any real district.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src import anomaly, charts, data_gen, evaluate, forecast, fund_balance, scenarios, staffing, variance

ROOT = os.path.dirname(os.path.abspath(__file__))


def main() -> dict:
    data = data_gen.generate()
    os.makedirs(f"{ROOT}/data/synthetic", exist_ok=True)
    os.makedirs(f"{ROOT}/data/processed", exist_ok=True)
    os.makedirs(f"{ROOT}/reports/charts", exist_ok=True)
    for name, df in data.items():
        df.to_csv(f"{ROOT}/data/synthetic/{name}.csv", index=False)

    var = variance.analyze(data["expenditure"])
    fc = forecast.forecast_all(data["revenue"], data["expenditure"], data_gen.FORECAST_YEARS)
    score = fund_balance.sustainability_score(data["fund"], data["revenue"])
    proj_fund = fund_balance.project_fund_balance(data["fund"], fc["projection"])
    staff = staffing.analyze(data["staffing"])
    scen = scenarios.run_scenarios(data["revenue"], data["expenditure"],
                                   fc["projection"], data["fund"])
    anom = anomaly.detect(data["expenditure"])
    chart_paths = charts.build_all(data, fc, scen, proj_fund, f"{ROOT}/reports/charts")

    metrics = evaluate.build(var, fc, score, proj_fund, staff, scen, anom)
    evaluate.write(metrics, f"{ROOT}/reports/metrics.json")
    var["by_dept"].to_csv(f"{ROOT}/data/processed/variance_by_department.csv", index=False)
    scen.to_csv(f"{ROOT}/data/processed/scenarios.csv", index=False)
    proj_fund.to_csv(f"{ROOT}/data/processed/projected_fund_balance.csv", index=False)

    print("=== School District Budget Forecasting & Fiscal Sustainability ===")
    print(f"Variance: mean |var| {metrics['variance']['mean_abs_variance_pct']:.2f}% | "
          f"flagged >5%: {metrics['variance']['n_flagged_over_5pct']}")
    print("Forecast holdout MAPE (revenue): "
          + ", ".join(f"{k} {v:.2f}%" for k, v in
                      metrics["forecast_holdout_mape_pct"]["revenue"].items()))
    print("Forecast holdout MAPE (expenditure): "
          + ", ".join(f"{k} {v:.2f}%" for k, v in
                      metrics["forecast_holdout_mape_pct"]["expenditure"].items()))
    print(f"Sustainability score: {score['score']}/100 ({score['band']})")
    print(f"Anomaly detection — rule F1 {anom['rule']['f1']:.3f} | "
          f"IsolationForest F1 {anom['isolation_forest']['f1']:.3f}")
    print(f"Charts: {len(chart_paths)} written to reports/charts/")
    print("Metrics written to reports/metrics.json")
    return metrics


if __name__ == "__main__":
    main()
