"""Fund-balance trajectory and fiscal-sustainability scoring.

Score (0-100) is a RULE-BASED composite, stated as a judgement framework,
not a statistical model:
  - 40 pts: unassigned fund balance as % of next-year budget
    (>=4% full marks — the commonly cited NY school-district guidepost;
     2-4% partial; <2% zero). [ASSUMPTION: the 4% guidepost is a planning
     rule of thumb, not a legal requirement.]
  - 25 pts: structural balance — 3-year average operating surplus/deficit
    as % of revenue (>=0 full marks, < -2% zero).
  - 20 pts: revenue diversity — Herfindahl-style concentration of the
    four revenue sources (lower concentration = higher score).
  - 15 pts: reserve trend — fund balance rising over the last 3 years.
Bands: >=70 Sound, 50-69 Watch, <50 At Risk.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

REVENUE_SOURCES = ["Property Tax Levy", "State Aid", "Federal Aid", "Other Local Revenue"]


def _clip(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


def sustainability_score(fund: pd.DataFrame, revenue: pd.DataFrame) -> dict:
    last3 = fund.tail(3)
    fb_pct = float(fund.iloc[-1]["fund_balance_pct_of_next_budget"])
    fb_score = 40 * _clip(fb_pct / 4.0)

    struct = float((last3["operating_surplus_deficit"] / last3["total_revenue"]).mean() * 100)
    struct_score = 25 * _clip((struct + 2.0) / 2.0)

    shares = revenue.iloc[-1][REVENUE_SOURCES].to_numpy(dtype=float)
    shares = shares / shares.sum()
    hhi = float((shares ** 2).sum())  # 0.25 (perfectly spread) to 1.0
    div_score = 20 * _clip((1.0 - hhi) / 0.75)

    trend = float(np.polyfit(np.arange(3), last3["fund_balance_eoy"].to_numpy(), 1)[0])
    trend_score = 15.0 if trend > 0 else (7.5 if abs(trend) < 1e-6 else 0.0)

    total = round(fb_score + struct_score + div_score + trend_score, 1)
    band = "Sound" if total >= 70 else ("Watch" if total >= 50 else "At Risk")
    return {
        "score": total,
        "band": band,
        "components": {
            "fund_balance_pct": round(fb_score, 1),
            "structural_balance": round(struct_score, 1),
            "revenue_diversity": round(div_score, 1),
            "reserve_trend": round(trend_score, 1),
        },
        "fund_balance_pct_of_budget": fb_pct,
        "structural_balance_pct": round(struct, 2),
        "revenue_hhi": round(hhi, 3),
        "reserve_trend_usd_per_year": round(trend, 0),
    }


def project_fund_balance(fund: pd.DataFrame, projection: pd.DataFrame) -> pd.DataFrame:
    """Roll the fund balance forward on projected surpluses/deficits."""
    balance = float(fund.iloc[-1]["fund_balance_eoy"])
    rows = []
    for _, row in projection.iterrows():
        balance += float(row["projected_surplus_deficit"])
        rows.append({
            "fiscal_year": int(row["fiscal_year"]),
            "projected_surplus_deficit": round(float(row["projected_surplus_deficit"]), 2),
            "projected_fund_balance_eoy": round(balance, 2),
            "projected_fund_balance_pct_of_budget": round(
                100 * balance / float(row["projected_expenditure"]), 2),
        })
    return pd.DataFrame(rows)
