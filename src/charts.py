"""Charts for the report and deck (matplotlib, Agg backend)."""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams.update({"font.size": 9, "figure.dpi": 150})


def _save(fig, path):
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def revenue_vs_expenditure(fund, path):
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    x = fund["fiscal_year"]
    ax.plot(x, fund["total_revenue"] / 1e6, marker="o", label="Total revenue")
    ax.plot(x, fund["total_expenditure"] / 1e6, marker="s", label="Total expenditure")
    ax.set_title("Revenue vs expenditure — history (synthetic, $M)")
    ax.set_xlabel("Fiscal year"); ax.set_ylabel("$ millions")
    ax.legend(); ax.grid(alpha=0.3)
    _save(fig, path)


def fund_balance(fund, projected, path):
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    x = list(fund["fiscal_year"]) + list(projected["fiscal_year"])
    y = list(fund["fund_balance_eoy"] / 1e6) + list(projected["projected_fund_balance_eoy"] / 1e6)
    ax.plot(x, y, marker="o", color="#1f77b4")
    ax.axhline(0, color="red", linestyle="--", linewidth=1)
    ax.set_title("Fund balance, end of year — history + base-case projection ($M)")
    ax.set_xlabel("Fiscal year"); ax.set_ylabel("$ millions")
    ax.grid(alpha=0.3)
    _save(fig, path)


def variance_heatmap(expenditure, path):
    piv = expenditure.pivot(index="department", columns="fiscal_year",
                            values="variance_pct") * 100
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    im = ax.imshow(piv.values, cmap="RdYlGn", vmin=-8, vmax=8, aspect="auto")
    ax.set_xticks(range(len(piv.columns))); ax.set_xticklabels(piv.columns)
    ax.set_yticks(range(len(piv.index))); ax.set_yticklabels(piv.index)
    ax.set_title("Budget variance % by department and year (red = over budget)")
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            ax.text(j, i, f"{piv.values[i, j]:+.1f}", ha="center", va="center", fontsize=7)
    fig.colorbar(im, ax=ax, label="variance %")
    _save(fig, path)


def forecast_holdout(forecasts, path):
    fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.6), sharey=False)
    for ax, (name, res) in zip(axes, [("Revenue", forecasts["revenue"]),
                                       ("Expenditure", forecasts["expenditure"])]):
        yrs = res["holdout_years"]
        ax.plot(yrs, np.array(res["actual"]) / 1e6, marker="o", label="Actual")
        for k, style in [("naive", "--"), ("linear_trend", "-."),
                         ("ridge", ":")]:
            ax.plot(yrs, np.array(res[k]["pred"]) / 1e6, marker="x",
                    linestyle=style, label=f"{k} (MAPE {res[k]['mape']:.2f}%)")
        ax.set_title(f"{name} — holdout FY2023-25 ($M)")
        ax.set_xlabel("Fiscal year"); ax.legend(fontsize=7); ax.grid(alpha=0.3)
    _save(fig, path)


def scenario_tornado(scenarios, path):
    df = scenarios.copy()
    base = df.loc[df["scenario"] == "Base case", "fy28_fund_balance"].iloc[0]
    df = df[df["scenario"] != "Base case"].copy()
    df["delta"] = (df["fy28_fund_balance"] - base) / 1e6
    df = df.sort_values("delta")
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    colors = ["#c0392b" if v < 0 else "#27ae60" for v in df["delta"]]
    ax.barh(df["scenario"], df["delta"], color=colors)
    ax.axvline(0, color="black", linewidth=1)
    ax.set_title("Scenario impact on FY2028 fund balance vs base case ($M)")
    ax.set_xlabel("Change in FY2028 fund balance ($M)")
    ax.grid(alpha=0.3, axis="x")
    _save(fig, path)


def staffing_costs(staffing, path):
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    x = staffing["fiscal_year"]
    ax.plot(x, staffing["salary_cost"] / 1e6, marker="o", label="Salary cost")
    ax.plot(x, staffing["benefit_cost"] / 1e6, marker="s", label="Benefit cost")
    ax.set_title("Staffing costs — salary vs benefits ($M)")
    ax.set_xlabel("Fiscal year"); ax.set_ylabel("$ millions")
    ax.legend(); ax.grid(alpha=0.3)
    _save(fig, path)


def build_all(data, forecasts, scenarios_df, projected_fund, outdir):
    import os
    os.makedirs(outdir, exist_ok=True)
    paths = {}
    paths["revenue_vs_expenditure"] = f"{outdir}/revenue_vs_expenditure.png"
    revenue_vs_expenditure(data["fund"], paths["revenue_vs_expenditure"])
    paths["fund_balance"] = f"{outdir}/fund_balance.png"
    fund_balance(data["fund"], projected_fund, paths["fund_balance"])
    paths["variance_heatmap"] = f"{outdir}/variance_heatmap.png"
    variance_heatmap(data["expenditure"], paths["variance_heatmap"])
    paths["forecast_holdout"] = f"{outdir}/forecast_holdout.png"
    forecast_holdout(forecasts, paths["forecast_holdout"])
    paths["scenario_tornado"] = f"{outdir}/scenario_tornado.png"
    scenario_tornado(scenarios_df, paths["scenario_tornado"])
    paths["staffing_costs"] = f"{outdir}/staffing_costs.png"
    staffing_costs(data["staffing"], paths["staffing_costs"])
    return paths
