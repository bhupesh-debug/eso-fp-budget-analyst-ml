"""Scenario engine: state-aid, enrollment and tax-cap shocks.

Each scenario re-runs the FY2026-28 projection from the base forecast by
adjusting the revenue side (state aid, levy) or the expenditure side
(enrollment-linked costs, benefits). Outputs the FY2028 fund balance and
sustainability band per scenario, plus a sensitivity ranking.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

SCENARIOS = {
    "Base case": {},
    "State aid -5%": {"state_aid_pct": -0.05},
    "State aid +5%": {"state_aid_pct": 0.05},
    "Enrollment -3%": {"enrollment_pct": -0.03},
    "Enrollment +3%": {"enrollment_pct": 0.03},
    "Tax levy frozen (0% growth)": {"levy_growth": 0.0},
    "Benefits +2pp growth": {"benefit_growth_pp": 0.02},
}


def run_scenarios(revenue: pd.DataFrame, expenditure: pd.DataFrame,
                  projection: pd.DataFrame, fund: pd.DataFrame) -> pd.DataFrame:
    last_rev = revenue.iloc[-1]
    base_rev = float(last_rev["Total Revenue"])
    base_exp = float(
        expenditure.loc[expenditure["fiscal_year"] == expenditure["fiscal_year"].max(),
                        "actual_expenditure"].sum())
    base_balance = float(fund.iloc[-1]["fund_balance_eoy"])
    # Base-case annual surplus from the projection.
    base_surplus = projection["projected_surplus_deficit"].to_numpy(dtype=float)

    rows = []
    for name, adj in SCENARIOS.items():
        surplus = base_surplus.copy()
        # State-aid shock: ~40% of revenue is state aid; shock applies each year.
        if "state_aid_pct" in adj:
            shock = float(last_rev["State Aid"]) * adj["state_aid_pct"]
            surplus = surplus + shock
        # Levy freeze: levy grows 1.9%/yr in base; freezing loses that growth,
        # compounding over the 3-year horizon.
        if "levy_growth" in adj:
            levy = float(last_rev["Property Tax Levy"])
            lost = sum(levy * (1.019 ** k) * 0.019 for k in range(1, 4))
            # Spread the cumulative loss across the 3 years (approx. by year).
            yearly = [levy * 0.019 * (1.019 ** k) for k in range(1, 4)]
            surplus = surplus - np.array(yearly)
        # Enrollment shock: state aid is enrollment-driven (foundation aid is
        # per-pupil, elasticity ~1.0) while costs are sticky in the short run
        # (elasticity ~0.3 — you cannot cut a teacher for 3% fewer pupils).
        # Net: an enrollment decline HURTS the balance.
        if "enrollment_pct" in adj:
            d = adj["enrollment_pct"]
            surplus = surplus - base_exp * 0.3 * d + float(last_rev["State Aid"]) * 1.0 * d
        # Benefits shock: benefits are ~12% of spend; +2pp growth adds cost.
        if "benefit_growth_pp" in adj:
            benefit_base = base_exp * 0.12
            yearly = [benefit_base * adj["benefit_growth_pp"] * k for k in (1, 2, 3)]
            surplus = surplus - np.array(yearly)
        balance = base_balance + float(surplus.sum())
        rows.append({
            "scenario": name,
            "fy26_28_cumulative_surplus": round(float(surplus.sum()), 0),
            "fy28_fund_balance": round(balance, 0),
            "fy28_fund_balance_pct_of_budget": round(
                100 * balance / float(projection.iloc[-1]["projected_expenditure"]), 2),
        })
    return pd.DataFrame(rows)
