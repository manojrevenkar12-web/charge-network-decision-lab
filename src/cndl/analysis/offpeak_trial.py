"""Question 2: should the night discount (30% off from 22:00 to 07:00) go network-wide?

The 2026 trial randomised the discount by site and day (a switchback) on 80 high-power sites for
eight weeks. Randomising days within sites removes the selection problem of question 1, but two
things still need care: yesterday's discount can shift today's charging (carryover), and daily
revenue is noisy, so the design needs enough site-days and a good covariate.

The analysis designs the trial from pre-trial data (minimum detectable effect, with and without
CUPED), then estimates the discount's effect on revenue, on the share of energy charged at night
and on daytime sessions (the queues it relieves), with the previous day's discount in the model,
site and date fixed effects and standard errors clustered by site, confirms the revenue effect
with a randomisation test, and shows when an always-valid sequential test would have stopped.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pyfixest as pf
from scipy import stats

from ..settings import analysis
from ..warehouse import connect

OUTCOMES = {
    "revenue_eur": "daily revenue per site (EUR)",
    "offpeak_share": "share of energy charged at night",
    "log_peak_sessions": "daytime sessions (log)",
}


def load() -> pd.DataFrame:
    con = connect()
    df = con.execute("select * from trial_day order by site_id, date").df()
    con.close()
    df["date"] = pd.to_datetime(df.date)
    df["log_peak_sessions"] = np.log(df.peak_sessions.clip(lower=1))
    df["dow"] = df.date.dt.dayofweek
    # CUPED covariate: the same site's average on the same weekday before the trial
    pre = df[~df.in_trial].groupby(["site_id", "dow"])[list(OUTCOMES)].mean().add_suffix("_pre")
    df = df.join(pre, on=["site_id", "dow"])
    trial = df[df.in_trial].copy()
    trial["discount"] = trial.discount.astype(int)
    trial["prev_discount"] = trial.groupby("site_id").discount.shift(1).fillna(0).astype(int)
    return trial


def _fit(d: pd.DataFrame, y: str) -> Any:
    data = (
        d[[y, "discount", "prev_discount", "site_id", "date"]]
        .dropna()
        .assign(date=lambda x: x.date.dt.strftime("%Y-%m-%d"))
    )
    return pf.feols(f"{y} ~ discount + prev_discount | site_id + date", data=data, vcov={"CRV1": "site_id"})


def estimate(trial: pd.DataFrame, alpha: float) -> dict[str, Any]:
    out = {}
    for y, label in OUTCOMES.items():
        fit = _fit(trial, y)
        coef, ses = fit.coef(), fit.se()
        b, se = float(coef["discount"]), float(ses["discount"])
        z = stats.norm.ppf(1 - alpha / 2)
        out[y] = {
            "label": label,
            "estimate": b,
            "ci": [b - z * se, b + z * se],
            "carryover": float(coef["prev_discount"]),
            "carryover_ci": [
                float(coef["prev_discount"] - z * ses["prev_discount"]),
                float(coef["prev_discount"] + z * ses["prev_discount"]),
            ],
            "control_mean": float(trial.loc[trial.discount == 0, y].mean()),
        }
    return out


def randomisation_p(trial: pd.DataFrame, y: str, reps: int, seed: int = 21) -> float:
    """Re-draw the coin flips (and so yesterday's flips) and compare the within-site difference."""
    rng = np.random.default_rng(seed)

    def stat(d: pd.DataFrame) -> float:
        g = d.groupby(["site_id", "discount"])[y].mean().unstack()
        return float((g[1] - g[0]).mean())

    observed = stat(trial)
    more = 0
    d = trial[["site_id", y]].copy()
    for _ in range(reps):
        d["discount"] = rng.random(len(d)) < 0.5
        d["discount"] = d.discount.astype(int)
        if abs(stat(d)) >= abs(observed):
            more += 1
    return (more + 1) / (reps + 1)


def design(alpha: float, sims: int, seed: int = 4) -> dict[str, Any]:
    """Minimum detectable revenue effect (80% power) for the trial's size, from pre-trial days.

    A/A simulations on the eight weeks before the trial: random switchback assignments, the same
    estimator, with and without the CUPED covariate (pre-period same-weekday mean)."""
    con = connect()
    pre = con.execute("select * from trial_day where not in_trial order by site_id, date").df()
    con.close()
    pre["date"] = pd.to_datetime(pre.date)
    pre["dow"] = pre.date.dt.dayofweek
    half = pre.date.min() + (pre.date.max() - pre.date.min()) / 2
    # covariate from the first half, simulated trial on the second half
    cov = pre[pre.date < half].groupby(["site_id", "dow"]).revenue_eur.mean().rename("revenue_eur_pre")
    d = pre[pre.date >= half].join(cov, on=["site_id", "dow"]).dropna()
    rng = np.random.default_rng(seed)
    ses = {"plain": [], "cuped": []}
    for _ in range(sims):
        d["discount"] = (rng.random(len(d)) < 0.5).astype(int)
        for kind in ses:
            y = (
                d.revenue_eur - (d.revenue_eur_pre - d.revenue_eur_pre.mean()) * _theta(d)
                if kind == "cuped"
                else d.revenue_eur
            )
            diff = y[d.discount == 1].mean() - y[d.discount == 0].mean()
            ses[kind].append(diff)
    days_sim = d.date.nunique()
    trial_days = 7 * analysis()["offpeak_trial"]["weeks"]
    scale = np.sqrt(days_sim / trial_days)
    z = stats.norm.ppf(1 - alpha / 2) + stats.norm.ppf(0.8)
    sd_plain = float(np.std(ses["plain"])) * scale
    sd_cuped = float(np.std(ses["cuped"])) * scale
    return {
        "sites": int(d.site_id.nunique()),
        "days": int(trial_days),
        "mde_revenue_eur_plain": z * sd_plain,
        "mde_revenue_eur_cuped": z * sd_cuped,
        "variance_reduction": 1 - (sd_cuped / sd_plain) ** 2 if sd_plain > 0 else 0.0,
        "mean_daily_revenue_eur": float(d.revenue_eur.mean()),
    }


def _theta(d: pd.DataFrame) -> float:
    c = np.cov(d.revenue_eur, d.revenue_eur_pre)
    return float(c[0, 1] / c[1, 1])


def sequential(trial: pd.DataFrame, y: str, alpha: float, tau: float) -> dict[str, Any]:
    """Mixture SPRT on the running within-site difference (always-valid p-values), day by day."""
    days = sorted(trial.date.unique())
    stop_day = None
    path = []
    for k in range(7, len(days) + 1):
        d = trial[trial.date.isin(days[:k])]
        g = d.groupby(["site_id", "discount"])[y].mean().unstack().dropna()
        diffs = (g[1] - g[0]).to_numpy()
        n = len(diffs)
        m = diffs.mean()
        v = diffs.var(ddof=1) / n
        # normal mixture likelihood ratio with prior N(0, tau^2) on the effect
        lr = np.sqrt(v / (v + tau**2)) * np.exp(m**2 * tau**2 / (2 * v * (v + tau**2)))
        p = float(min(1.0, 1 / lr))
        path.append({"day": k, "estimate": float(m), "always_valid_p": p})
        if stop_day is None and p < alpha:
            stop_day = k
    return {"stopped_on_day": stop_day, "path": path}


def run() -> dict[str, Any]:
    cfg = analysis()["offpeak_trial"]
    trial = load()
    est = estimate(trial, cfg["alpha"])
    p_rand = randomisation_p(trial, "revenue_eur", 499)
    seq = sequential(trial, "revenue_eur", cfg["alpha"], tau=est["revenue_eur"]["control_mean"] * 0.05)
    des = design(cfg["alpha"], cfg["power_simulations"])
    # the decision: revenue given up against the value of relieved daytime demand
    peak_base = float(np.exp(trial.loc[trial.discount == 0, "log_peak_sessions"]).mean())
    rel = est["log_peak_sessions"]
    relieved = [
        -peak_base * np.expm1(rel["ci"][1]),
        -peak_base * np.expm1(rel["estimate"]),
        -peak_base * np.expm1(rel["ci"][0]),
    ]
    value = cfg["peak_session_value_eur"]
    rev = est["revenue_eur"]
    net = [
        rev["ci"][0] + value * relieved[0],
        rev["estimate"] + value * relieved[1],
        rev["ci"][1] + value * relieved[2],
    ]
    return {
        "question": "Should the night discount go network-wide?",
        "sites": int(trial.site_id.nunique()),
        "site_days": len(trial),
        "design": des,
        "effects": est,
        "randomisation_p_revenue": p_rand,
        "sequential": seq,
        "peak_sessions_relieved_per_site_day": relieved,
        "peak_session_value_eur": value,
        "net_value_per_site_day": net,
        "decision": "roll out" if net[0] > 0 else ("do not roll out" if net[2] < 0 else "extend the trial"),
    }
