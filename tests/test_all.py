"""Test suite — prints a runtime "N passed" line.

Run: python -m tests.test_all
The count comes from this runtime counter, never from counting check()
call sites (checks inside loops execute multiple times).
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from src import anomaly, data_gen, forecast, fund_balance, scenarios, staffing, variance

PASSED = 0
FAILED = 0
FAILURES = []


def check(name: str, cond: bool, detail: str = ""):
    global PASSED, FAILED
    if cond:
        PASSED += 1
    else:
        FAILED += 1
        FAILURES.append(f"{name}: {detail}")


def main():
    global PASSED, FAILED
    data = data_gen.generate()
    data2 = data_gen.generate()
    check("data_gen deterministic (revenue identical across runs)",
          data["revenue"].equals(data2["revenue"]))
    check("revenue has 8 history years", len(data["revenue"]) == 8)
    check("expenditure has 8 departments x 8 years",
          len(data["expenditure"]) == 64)
    check("fund balance positive in final history year",
          float(data["fund"].iloc[-1]["fund_balance_eoy"]) > 0)
    check("total revenue equals sum of sources",
          np.allclose(data["revenue"]["Total Revenue"],
                      data["revenue"][data_gen.REVENUE_SOURCES].sum(axis=1)))
    check("exactly 3 injected anomalies", len(data["anomalies"]) == 3)
    check("enrollment declines over history",
          float(data["revenue"]["enrollment"].iloc[-1])
          < float(data["revenue"]["enrollment"].iloc[0]))
    check("federal aid bump present FY2021-22 then rolls off",
          float(data["revenue"].loc[data["revenue"]["fiscal_year"] == 2021, "Federal Aid"].iloc[0])
          > float(data["revenue"].loc[data["revenue"]["fiscal_year"] == 2023, "Federal Aid"].iloc[0]))

    var = variance.analyze(data["expenditure"])
    check("mean abs variance between 0 and 10%",
          0 < var["mean_abs_variance_pct"] < 0.10)
    check("flagged rows are a subset of all rows",
          0 <= var["n_flagged"] <= len(data["expenditure"]))
    check("variance by year has 8 rows", len(var["by_year"]) == 8)
    check("variance by dept has 8 rows", len(var["by_dept"]) == 8)
    check("worst department is a real department",
          var["by_dept"].iloc[0]["department"] in data_gen.DEPARTMENTS)
    check("by-year variance recomputes from detail (tolerance for 4-dp rounding)",
          np.allclose(
              var["by_year"]["variance_pct"].to_numpy(),
              ((var["by_year"]["total_actual"] - var["by_year"]["total_adopted"])
               / var["by_year"]["total_adopted"]).to_numpy(), atol=1e-4))

    fc = forecast.forecast_all(data["revenue"], data["expenditure"], [2026, 2027, 2028])
    for target in ("revenue", "expenditure"):
        for model in ("naive", "linear_trend", "ridge"):
            mape = fc[target][model]["mape"]
            check(f"{target} {model} MAPE finite and < 25%",
                  np.isfinite(mape) and 0 <= mape < 25, f"mape={mape}")
        check(f"{target} best model is one of the three",
              fc[target]["best"] in ("naive", "linear_trend", "ridge"))
        check(f"{target} holdout has 3 years", len(fc[target]["holdout_years"]) == 3)
    check("projection has 3 years", len(fc["projection"]) == 3)
    check("projected revenue positive", bool((fc["projection"]["projected_revenue"] > 0).all()))
    check("projected expenditure positive",
          bool((fc["projection"]["projected_expenditure"] > 0).all()))
    # Best-model selection is consistent: best == argmin of the three MAPEs.
    # (No assumption about WHICH model wins — on revenue the naive model wins
    # because the federal-aid cliff breaks every trend; on expenditure the
    # linear trend wins. That asymmetry is the finding.)
    for target in ("revenue", "expenditure"):
        mapes = {k: fc[target][k]["mape"] for k in ("naive", "linear_trend", "ridge")}
        check(f"{target} best model equals argmin MAPE",
              fc[target]["best"] == min(mapes, key=mapes.get))

    score = fund_balance.sustainability_score(data["fund"], data["revenue"])
    check("sustainability score in [0, 100]", 0 <= score["score"] <= 100)
    check("sustainability band valid", score["band"] in ("Sound", "Watch", "At Risk"))
    check("score components sum to total",
          abs(sum(score["components"].values()) - score["score"]) < 0.6)
    proj = fund_balance.project_fund_balance(data["fund"], fc["projection"])
    check("projected fund balance has 3 rows", len(proj) == 3)
    check("projected fund balance rolls forward arithmetically",
          abs(float(proj.iloc[-1]["projected_fund_balance_eoy"])
              - (float(data["fund"].iloc[-1]["fund_balance_eoy"])
                 + float(fc["projection"]["projected_surplus_deficit"].sum()))) < 1.0)

    staff = staffing.analyze(data["staffing"])
    check("staffing R2 in [0, 1]", 0 <= staff["r2"] <= 1)
    check("staffing R2 above 0.8 (cost is near-linear in drivers)",
          staff["r2"] > 0.8, f"r2={staff['r2']}")
    check("benefit cost grows faster than salary (CAGR)",
          staff["benefit_cost_cagr_pct"] > staff["avg_salary_cagr_pct"])
    check("driver ranking has 3 drivers", len(staff["driver_ranking"]) == 3)
    check("per-pupil staffing cost rises over history",
          staff["per_pupil_staffing_cost_last"] > staff["per_pupil_staffing_cost_first"])

    scen = scenarios.run_scenarios(data["revenue"], data["expenditure"],
                                   fc["projection"], data["fund"])
    check("7 scenarios produced", len(scen) == 7)
    check("base case present", "Base case" in scen["scenario"].tolist())
    check("state aid -5% worsens FY28 balance vs base",
          float(scen.loc[scen["scenario"] == "State aid -5%", "fy28_fund_balance"].iloc[0])
          < float(scen.loc[scen["scenario"] == "Base case", "fy28_fund_balance"].iloc[0]))
    check("state aid +5% improves FY28 balance vs base",
          float(scen.loc[scen["scenario"] == "State aid +5%", "fy28_fund_balance"].iloc[0])
          > float(scen.loc[scen["scenario"] == "Base case", "fy28_fund_balance"].iloc[0]))
    check("levy freeze worsens FY28 balance vs base",
          float(scen.loc[scen["scenario"] == "Tax levy frozen (0% growth)", "fy28_fund_balance"].iloc[0])
          < float(scen.loc[scen["scenario"] == "Base case", "fy28_fund_balance"].iloc[0]))
    check("enrollment -3% worsens FY28 balance vs base (sticky costs)",
          float(scen.loc[scen["scenario"] == "Enrollment -3%", "fy28_fund_balance"].iloc[0])
          < float(scen.loc[scen["scenario"] == "Base case", "fy28_fund_balance"].iloc[0]))

    anom = anomaly.detect(data["expenditure"])
    check("rule detector finds at least 1 of 3 injected anomalies",
          anom["rule"]["tp"] >= 1, f"tp={anom['rule']['tp']}")
    check("rule precision above 0.3 (not just flagging everything)",
          anom["rule"]["precision"] > 0.3, f"p={anom['rule']['precision']}")
    check("isolation forest flags a plausible count (1-10 of 64)",
          1 <= anom["iso_n_flagged"] <= 10, f"n={anom['iso_n_flagged']}")
    check("rule and forest F1 both finite",
          np.isfinite(anom["rule"]["f1"]) and np.isfinite(anom["isolation_forest"]["f1"]))

    print(f"{PASSED} passed, {FAILED} failed")
    if FAILURES:
        print("FAILURES:")
        for f in FAILURES:
            print(" -", f)
    sys.exit(1 if FAILED else 0)


if __name__ == "__main__":
    main()
