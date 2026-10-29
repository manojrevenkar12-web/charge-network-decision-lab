"""Question 1: what did the 2025 price increase (+€0.10/kWh) do to sessions and revenue?

Country managers raised prices in three waves, and chose the sites. If they chose sites that were
growing faster, comparing raised and unchanged sites confuses growth with the price effect. The
analysis therefore:

1. shows the estimates an analyst would usually reach for (before/after, two-way fixed effects,
   Gardner's two-stage DiD for staggered adoption),
2. tests the assumption they share, parallel trends, with an event study of the weeks before
   each site's increase,
3. if that test fails, estimates the effect under two weaker assumptions that bracket the truth
   from different sides: synthetic DiD (controls reweighted to grow like the treated sites) and
   trend imputation (each site would have kept its own growth path, with a capacity-aware
   transform because busy sites cannot keep growing linearly),
4. reports the range those credible estimates span, with bootstrap intervals, and what it means
   for revenue, because a range that the decision does not depend on is enough for a decision.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pyfixest as pf
from scipy import stats

from ..settings import analysis
from ..warehouse import connect
from . import impute, sdid

# what an analyst can know about charger capacity: sessions a point can serve per day
# (24 hours divided by a typical session plus changeover; not read from the simulator)
CAPACITY_PER_POINT = {"HPC": 36.0, "DC": 24.0, "AC": 9.0}


def load() -> pd.DataFrame:
    con = connect()
    df = con.execute(
        """
        select w.site_id, w.week, w.sessions, w.availability, w.price_eur_kwh,
               s.segment, s.country, s.charger_type, s.points, p.first_week
        from site_week w join sites s using (site_id) left join price_increase p using (site_id)
        -- a two-year window, and sites old enough that ramp-up after opening is over
        where w.week >= date '2024-01-01' and s.opened < date '2023-01-01'
        """
    ).df()
    con.close()
    df["week"] = pd.to_datetime(df.week)
    df["first_week"] = pd.to_datetime(df.first_week)
    df["y"] = np.log(df.sessions.clip(lower=1))
    df["lavail"] = np.log(df.availability.clip(lower=0.05))
    df["treat"] = (df.first_week.notna() & (df.week >= df.first_week)).astype(int)
    weeks = sorted(df.week.unique())
    wi = {w: i + 1 for i, w in enumerate(weeks)}
    df["t"] = df.week.map(wi)
    df["g"] = df.first_week.map(lambda x: wi.get(x, 0) if pd.notna(x) else 0).astype(int)
    df["wseg"] = df.week.dt.strftime("%Y-%m-%d") + df.segment
    return df


def _ci(draws: np.ndarray, level: float = 0.95) -> tuple[float, float]:
    lo, hi = np.percentile(draws, [50 * (1 - level), 100 - 50 * (1 - level)])
    return float(lo), float(hi)


def naive(df: pd.DataFrame) -> dict[str, Any]:
    """Treated sites, 26 weeks after their increase vs the 26 weeks before."""
    t = df[df.g > 0]
    rel = t.t - t.g
    after = t[(rel >= 0) & (rel < 26)].groupby("site_id").y.mean()
    before = t[(rel < 0) & (rel >= -26)].groupby("site_id").y.mean()
    d = (after - before).dropna()
    se = d.std(ddof=1) / np.sqrt(len(d))
    return {
        "method": "before_after",
        "estimate": float(d.mean()),
        "ci": [float(d.mean() - 1.96 * se), float(d.mean() + 1.96 * se)],
    }


def twfe(df: pd.DataFrame) -> dict[str, Any]:
    fit = pf.feols("y ~ treat + lavail | site_id + wseg", data=df, vcov={"CRV1": "site_id"})
    tid = fit.tidy().loc["treat"]
    return {"method": "twfe", "estimate": float(tid["Estimate"]), "ci": [float(tid["2.5%"]), float(tid["97.5%"])]}


def _panel(df: pd.DataFrame, y: str = "y") -> impute.Panel:
    return impute.to_panel(df, y, "segment", "lavail")


def imputation(df: pd.DataFrame, reps: int) -> dict[str, Any]:
    """Imputation under parallel trends (Borusyak, Jaravel and Spiess 2024; the same point estimate
    as Gardner's two-stage DiD): site effects and segment-by-week effects fitted on untreated weeks."""
    p = _panel(df)

    def est(rows: np.ndarray | None) -> float:
        rows = np.arange(len(p.sites)) if rows is None else rows
        y0, usable = impute.fit_untreated(p, rows, trend=False)
        m = p.treated[rows] & ~np.isnan(p.y[rows]) & usable[:, None]
        return float(np.mean((p.y[rows] - y0)[m]))

    rng = np.random.default_rng(3)
    n = len(p.sites)
    draws = np.array([est(rng.integers(0, n, n)) for _ in range(reps)])
    return {"method": "imputation_parallel_trends", "estimate": est(None), "ci": list(_ci(draws))}


def event_study(df: pd.DataFrame, leads: int, lags: int, reps: int) -> dict[str, Any]:
    """Effects by week relative to each site's increase, under parallel trends. The leads are
    placebo effects: with parallel trends they scatter around zero; a slope in them is a warning."""
    p = _panel(df)
    wi = {w: i for i, w in enumerate(p.weeks)}
    first = {s: wi[g] for s, g in df.drop_duplicates("site_id").set_index("site_id").first_week.dropna().items()}
    point = impute.event_time_effects(p, first, leads, lags)
    rng = np.random.default_rng(9)
    n = len(p.sites)
    boots = [
        impute.event_time_effects(p, first, leads, lags, rows=rng.integers(0, n, n)).set_index("event_week").effect
        for _ in range(reps)
    ]
    B = pd.concat(boots, axis=1).reindex(point.event_week)
    lo, hi = np.nanpercentile(B.to_numpy(), [2.5, 97.5], axis=1)
    coefs = pd.DataFrame({"event_week": point.event_week, "estimate": point.effect, "lo": lo, "hi": hi})

    # slope of the leads, and its bootstrap distribution
    def slope(series: pd.Series) -> float:
        pre = series[series.index < 0]
        return float(np.polyfit(pre.index.to_numpy(dtype=float), pre.to_numpy(), 1)[0])

    s0 = slope(point.set_index("event_week").effect)
    sb = np.array([slope(b.dropna()) for b in boots])
    se = float(np.std(sb, ddof=1))
    p_val = float(2 * (1 - stats.norm.cdf(abs(s0 / se)))) if se > 0 else 1.0
    return {
        "coefficients": coefs.round(5).to_dict("records"),
        "pretrend_slope_per_week": s0,
        "pretrend_slope_se": se,
        "pretrend_p": p_val,
    }


def synthetic_did(df: pd.DataFrame, reps: int) -> dict[str, Any]:
    wide = df.pivot(index="site_id", columns="week", values="y")
    full = wide.dropna()
    weeks = list(full.columns)
    fw = df.drop_duplicates("site_id").set_index("site_id").first_week.reindex(full.index)
    first = np.array([weeks.index(x) if pd.notna(x) else -1 for x in fw])
    Y = full.to_numpy()
    est, cohorts = sdid.staggered(Y, first)
    draws = sdid.bootstrap(Y, first, reps)
    return {
        "method": "synthetic_did",
        "estimate": est,
        "ci": list(_ci(draws)),
        "cohorts": {str(weeks[g].date()): {"estimate": e, "treated_site_weeks": n} for g, (e, n) in cohorts.items()},
        "sites": len(full),
    }


def trend_imputation(df: pd.DataFrame, reps: int) -> dict[str, Any]:
    """Each site keeps its own growth path, on a scale where capacity limits are linearised."""
    cap = df.charger_type.map(CAPACITY_PER_POINT) * df.points * 7
    avail = df.availability.clip(lower=0.3)
    s = np.minimum(df.sessions / avail, 0.97 * cap)
    d = df.assign(z=np.log(-cap * np.log(1 - s / cap)), cap=cap, avail=avail)
    pz = impute.to_panel(d, "z", "segment")
    C = impute.to_panel(d, "cap", "segment").y
    A = impute.to_panel(d, "avail", "segment").y
    Y = impute.to_panel(d, "y", "segment").y

    def estimate(rows: np.ndarray | None) -> float:
        rows = np.arange(len(pz.sites)) if rows is None else rows
        z0, usable = impute.fit_untreated(pz, rows)
        y0 = np.log(C[rows] * (1 - np.exp(-np.exp(z0) / C[rows])) * A[rows])
        m = pz.treated[rows] & ~np.isnan(Y[rows]) & usable[:, None]
        return float(np.mean((Y[rows] - y0)[m]))

    est = estimate(None)
    rng = np.random.default_rng(5)
    n = len(pz.sites)
    draws = np.array([estimate(rng.integers(0, n, n)) for _ in range(reps)])
    # effects by segment, under the same assumption
    z0, usable = impute.fit_untreated(pz)
    y0 = np.log(C * (1 - np.exp(-np.exp(z0) / C)) * A)
    m = pz.treated & ~np.isnan(Y) & usable[:, None]
    seg = df.drop_duplicates("site_id").set_index("site_id").segment.reindex(pz.sites).to_numpy()
    eff = Y - y0
    by_segment = {str(sg): float(np.mean(eff[(seg == sg)[:, None] & m])) for sg in np.unique(seg[m.any(axis=1)])}
    return {"method": "trend_imputation", "estimate": est, "ci": list(_ci(draws)), "by_segment": by_segment}


def run() -> dict[str, Any]:
    cfg = analysis()["price_change"]
    df = load()
    treated_sites = sorted(df.loc[df.treat == 1, "site_id"].unique())
    usual = [naive(df), twfe(df), imputation(df, cfg["bootstrap"] // 3)]
    es = event_study(df, cfg["pre_weeks"], cfg["post_weeks"], cfg["bootstrap"] // 3)
    trends_fail = es["pretrend_p"] < cfg["pretrend_alpha"]
    robust = [synthetic_did(df, cfg["bootstrap"] // 3), trend_imputation(df, cfg["bootstrap"])]
    credible = robust if trends_fail else [usual[2], *robust]
    lo = min(c["ci"][0] for c in credible)
    hi = max(c["ci"][1] for c in credible)
    point_lo = min(c["estimate"] for c in credible)
    point_hi = max(c["estimate"] for c in credible)
    # revenue: price up by delta on each treated site's old price; energy per session unchanged
    old = df[(df.g > 0) & (df.t < df.g)].groupby("site_id").price_eur_kwh.last()
    price_up = float(np.log((old + cfg["delta_eur"]) / old).mean())
    return {
        "question": "What did the 2025 price increase do to sessions and revenue?",
        "treated_sites": len(treated_sites),
        "control_sites": int(df.loc[df.g == 0, "site_id"].nunique()),
        "analysed_sites": treated_sites,
        "estimand": "average change in log weekly sessions at raised sites, over the weeks since their increase",
        "usual_estimates": usual,
        "event_study": es,
        "parallel_trends_rejected": bool(trends_fail),
        "credible_estimates": credible,
        "range_log": [lo, hi],
        "range_point_log": [point_lo, point_hi],
        "range_pct": [float(np.expm1(lo)), float(np.expm1(hi))],
        "price_increase_log": price_up,
        "revenue_range_pct": [float(np.expm1(lo + price_up)), float(np.expm1(hi + price_up))],
        "decision": "keep" if np.expm1(hi + price_up) > 0 and np.expm1(lo + price_up) > 0 else "review",
    }
