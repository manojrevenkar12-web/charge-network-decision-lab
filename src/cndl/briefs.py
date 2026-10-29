"""Decision briefs: one page per question for people who decide, built from the analyses' results
(never from the truth). Each brief states the answer, the number with its uncertainty, the
decision, what the answer rests on, and what would change it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


def pct(x: float, digits: int = 1) -> str:
    return f"{x * 100:+.{digits}f}%"


def eur(x: float) -> str:
    a = abs(x)
    s = f"€{a / 1e6:,.0f}m" if a >= 1e6 else f"€{a:,.0f}"
    return f"-{s}" if x < 0 else s


def price_change(r: dict[str, Any]) -> dict[str, Any]:
    lo, hi = r["range_pct"]
    rlo, rhi = r["revenue_range_pct"]
    usual = {u["method"]: u["estimate"] for u in r["usual_estimates"]}
    es = r["event_study"]
    return {
        "id": "price-change",
        "title": "The 2025 price increase",
        "question": r["question"],
        "answer": f"It cost between {abs(hi) * 100:.0f}% and {abs(lo) * 100:.0f}% of sessions at the raised sites, and raised their revenue by {rlo * 100:.0f}% to {rhi * 100:.0f}%.",
        "number": {
            "label": "sessions lost",
            "value": float(np.mean(r["range_pct"])),
            "low": lo,
            "high": hi,
            "unit": "pct",
        },
        "decision": "Keep the higher price. Revenue rises at every effect size the data allows, so the decision does not depend on the exact number.",
        "rests_on": [
            f"{r['treated_sites']} raised sites against {r['control_sites']} unchanged ones over two years of weekly data.",
            f"The usual comparisons say sessions went up ({pct(usual['before_after'])} before and after, {pct(usual['twfe'])} with two-way fixed effects): the raised sites were growing faster.",
            f"The weeks before each increase show it: the raised sites pulled ahead by {es['pretrend_slope_per_week'] * 100:.2f}% a week before anything changed (p = {es['pretrend_p']:.1g}).",
            "Two methods that allow for that disagree in a known direction: controls reweighted to grow like the raised sites (synthetic DiD) give the smaller loss, each site's own growth continued (trend imputation) the larger one.",
        ],
        "would_change": [
            "A randomised price test, which would narrow the range to a single number (question 2 shows how).",
            "Evidence that growth at the raised sites was about to slow sharply, which would move the answer towards the smaller loss.",
        ],
        "method": "Before/after, two-way fixed effects, imputation (Borusyak, Jaravel and Spiess), event study, synthetic DiD (Arkhangelsky et al.), capacity-aware trend imputation; site bootstrap.",
    }


def trial(r: dict[str, Any]) -> dict[str, Any]:
    e = r["effects"]
    rev = e["revenue_eur"]
    net = r["net_value_per_site_day"]
    d = r["design"]
    return {
        "id": "night-discount",
        "title": "The night discount trial",
        "question": r["question"],
        "answer": (
            f"The 30% night discount moves {e['offpeak_share']['estimate'] * 100:.1f} points of energy to the night and cuts daytime sessions by "
            f"{-np.expm1(e['log_peak_sessions']['estimate']) * 100:.0f}%, but costs {eur(-rev['estimate'])} of revenue per site and day."
        ),
        "number": {
            "label": "net value per site and day",
            "value": net[1],
            "low": net[0],
            "high": net[2],
            "unit": "eur",
        },
        "decision": {
            "do not roll out": "Do not roll it out at 30%. The relieved daytime demand is worth less than the revenue it gives away; test a smaller discount on the busiest sites only.",
            "roll out": "Roll it out: the value of relieved daytime demand exceeds the revenue given away.",
            "extend the trial": "Extend the trial: the net value could still go either way.",
        }[r["decision"]],
        "rests_on": [
            f"{r['sites']} sites, {r['site_days']} site-days, discount assigned by coin flip each day (randomisation p = {r['randomisation_p_revenue']:.3f}).",
            f"Designed from pre-trial data: the trial could detect {eur(d['mde_revenue_eur_cuped'])} a day ({d['mde_revenue_eur_cuped'] / d['mean_daily_revenue_eur'] * 100:.1f}% of revenue) with 80% power.",
            f"Yesterday's discount still shifts a little charging today ({e['offpeak_share']['carryover'] * 100:.1f} points), so the model separates the two.",
            f"An always-valid sequential test would have stopped the trial on day {r['sequential']['stopped_on_day']}.",
        ],
        "would_change": [
            f"A higher value of a relieved daytime session than {eur(r['peak_session_value_eur'])} (grid peak charges, queues), which would have to exceed about {eur(-rev['estimate'] / max(r['peak_sessions_relieved_per_site_day'][1], 1e-9))} to pay.",
        ],
        "method": "Switchback design by site and day; fixed effects for site and date with yesterday's assignment; clustered standard errors; randomisation inference; CUPED for design; mixture SPRT.",
    }


def competitor(r: dict[str, Any]) -> dict[str, Any]:
    if not r.get("available"):
        return {
            "id": "competitor",
            "title": "A competitor next door",
            "question": r["question"],
            "answer": "No suitable case in the data.",
            "rests_on": [],
            "would_change": [],
        }
    p = r["pooled"]
    return {
        "id": "competitor",
        "title": "A competitor next door",
        "question": r["question"],
        "answer": (
            f"At {r['site_id']}, a competitor hub opened on {r['competitor_opened']} took about {abs(r['effect_pct']) * 100:.0f}% of sessions. "
            f"Across all {p['events']} openings in the history the loss is {abs(np.expm1(p['estimate_log'])) * 100:.0f}%."
        ),
        "number": {
            "label": "sessions lost at the site",
            "value": r["effect_pct"],
            "low": float(np.expm1(r["ci_log"][0])),
            "high": float(np.expm1(r["ci_log"][1])),
            "unit": "pct",
        },
        "decision": "Price the risk into site choices: a new site next to an existing charger should expect to lose about this share, which the rollout model includes.",
        "rests_on": [
            f"A synthetic control from {r['donors']} motorway high-power sites without a competitor or a price change, matching the site's path before the opening (pre-period error {r['pre_rmspe'] * 100:.1f}%).",
            f"Placebo test: the gap is larger than at {100 - r['placebo_p'] * 100:.0f}% of donor sites where nothing happened (p = {r['placebo_p']:.2f}).",
            "Openings fall on dates nobody at EDRI chose, so the network-wide estimate can assume parallel trends.",
        ],
        "would_change": ["A second competitor or a price war nearby, which the placebo sites would not reflect."],
        "method": "Synthetic control with an intercept and placebo inference; network-wide imputation estimator with site bootstrap.",
    }


def forecast(r: dict[str, Any]) -> dict[str, Any]:
    nxt, last = r["next_12_months_gwh"], r["last_12_months_gwh"]
    bt = r["backtest"][r["chosen"]]
    t = r["total"]
    lo = sum(x["lo"] for x in t) / 1e6
    hi = sum(x["hi"] for x in t) / 1e6
    names = {
        "capacity_aware": "the capacity-aware site model",
        "ets": "exponential smoothing",
        "seasonal_growth": "seasonal naive with growth",
        "global_gbm": "a gradient-boosted model over all sites",
    }
    return {
        "id": "forecast",
        "title": "Energy in the next twelve months",
        "question": r["question"],
        "answer": f"The existing network will sell about {nxt:,.0f} GWh in the next twelve months, {(nxt / last - 1) * 100:+.0f}% on the last twelve.",
        "number": {"label": "GWh, next 12 months", "value": nxt, "low": lo, "high": hi, "unit": "gwh"},
        "decision": "Plan energy purchasing and grid capacity on this figure; growth slows at the busiest sites as their chargers fill up.",
        "rests_on": [
            f"Chosen on a backtest of the last twelve months: {names[r['chosen']]} missed countries by {bt['mape_country'] * 100:.1f}% a month on average.",
            *[f"{names[k]}: {v['mape_country'] * 100:.1f}%." for k, v in r["backtest"].items() if k != r["chosen"]],
            f"The {r['interval'] * 100:.0f}% intervals come from the backtest's own errors at each horizon.",
        ],
        "would_change": [
            "New sites or price changes, which this forecast of the existing network at today's prices does not include."
        ],
        "method": "Seasonal naive, AutoETS (statsforecast), LightGBM global model (mlforecast), capacity-aware site trends; conformal intervals from a rolling backtest.",
    }


def rollout(r: dict[str, Any]) -> dict[str, Any]:
    p = r["plans"][r["recommended"]]
    g = r["plans"]["expected_npv_greedy"]
    t = r["plans"]["highest_traffic"]
    m = r["model"]
    top = ", ".join(f"{k} {v}" for k, v in sorted(p["by_country"].items(), key=lambda kv: -kv[1]))
    return {
        "id": "rollout",
        "title": "Where the next 1,000 points go",
        "question": r["question"],
        "answer": f"Build {p['sites']} sites with {p['points']} points ({top} sites), for {eur(p['capex_eur'])}: expected ten-year NPV {eur(p['expected_npv_eur'])}, and {eur(p['cvar_npv_eur'])} in the worst tenth of scenarios.",
        "number": {
            "label": "expected NPV",
            "value": p["expected_npv_eur"],
            "low": p["cvar_npv_eur"],
            "high": p["expected_npv_eur"],
            "unit": "eur",
        },
        "decision": f"Approve this plan. Choosing by traffic alone is expected to earn {eur(p['expected_npv_eur'] - t['expected_npv_eur'])} less; ranking by expected NPV without looking at sites that compete with each other, {eur(p['expected_npv_eur'] - g['expected_npv_eur'])} less.",
        "rests_on": [
            f"Demand from a Bayesian hierarchical model of {m['training_sites']} established sites (NUTS, r-hat {m['max_r_hat']:.2f}); unexplained site-to-site variation is {m['sigma']:.2f} on the log scale, which is most of the risk.",
            f"A nearby charging site costs demand (coefficient {m['pressure_coef']:.2f} per unit of proximity); the plan charges for new sites that would compete with each other.",
            "New sites priced like the 2025 increase, with its effect from question 1, and grown with question 4's country growth.",
            f"{r['candidates']} candidates, sizes of 4, 8 or 12 points, within the budget, with at least 18 Italian sites and at most a third of the points in one country.",
        ],
        "would_change": [
            "A different capex or energy price, which moves sizes before it moves sites.",
            "Grid connection quotes that differ from the planning figures.",
        ],
        "method": "NumPyro hierarchical model, posterior predictive scenarios, ten-year NPV per scenario, mixed-integer programme with CVaR (HiGHS).",
    }


def build(results: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        price_change(results["price_change"]),
        trial(results["trial"]),
        competitor(results["competitor"]),
        forecast(results["forecast"]),
        rollout(results["rollout"]),
    ]


def write_markdown(briefs: list[dict[str, Any]], out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    for b in briefs:
        lines = [f"# {b['title']}", "", f"**{b['question']}**", "", b["answer"], ""]
        if b.get("decision"):
            lines += ["## Decision", "", b["decision"], ""]
        lines += ["## What this rests on", "", *[f"- {x}" for x in b["rests_on"]], ""]
        lines += ["## What would change it", "", *[f"- {x}" for x in b["would_change"]], ""]
        if b.get("method"):
            lines += ["## Method", "", b["method"], ""]
        lines += ["_Generated by `cndl run` from the analyses' results._", ""]
        (out / f"{b['id']}.md").write_text("\n".join(lines))
