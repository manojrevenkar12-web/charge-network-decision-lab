"""Question 3: what does a competitor's hub next to one of our motorway sites cost us?

One site, one event: a difference-in-differences against "all other sites" would compare a busy
motorway hub with kerbside AC points. A synthetic control builds a weighted mix of comparable
sites (motorway, high-power, no competitor event and no price change around the date) whose
path before the opening matches the site's path, and reads the effect as the gap afterwards.
Inference is by placebo: the same method run on every donor as if it had been hit; the site's
effect is credible if its post/pre fit ratio stands out among them.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ..settings import analysis
from ..warehouse import connect
from . import impute
from .sdid import _fw_simplex

PHASE_IN_WEEKS = 8  # an analyst's choice: a new hub takes a couple of months to be found


def case() -> dict[str, Any] | None:
    """The most recent 2025 competitor opening next to an established motorway high-power site."""
    con = connect()
    row = con.execute(
        """
        select e.site_id, e.opened from competitor_events e join sites s using (site_id)
        where s.segment = 'motorway' and s.charger_type = 'HPC' and s.opened < date '2023-01-02'
          and e.opened between date '2025-01-01' and date '2025-12-31'
        order by e.site_id limit 1
        """
    ).fetchone()
    con.close()
    return {"site_id": row[0], "opened": pd.Timestamp(row[1])} if row else None


def _series(site: str, opened: pd.Timestamp, pre_weeks: int) -> tuple[pd.DataFrame, list[str]]:
    con = connect()
    start = opened - pd.Timedelta(weeks=pre_weeks)
    df = con.execute(
        """
        select w.site_id, w.week, ln(w.sessions) as y
        from site_week w join sites s using (site_id)
        where s.segment = 'motorway' and s.charger_type = 'HPC' and s.opened < date '2023-01-02'
          and w.week >= ?
          and (w.site_id = ? or (
                w.site_id not in (select site_id from competitor_events where opened >= ? - interval 26 week)
            and w.site_id not in (select site_id from price_increase)))
        """,
        [start.date(), site, start.date()],
    ).df()
    con.close()
    wide = df.pivot(index="site_id", columns="week", values="y").dropna()
    donors = [s for s in wide.index if s != site]
    return wide, donors


def _fit(wide: pd.DataFrame, target: str, donors: list[str], n_pre: int) -> dict[str, Any]:
    Y = wide.loc[donors].to_numpy()
    y = wide.loc[target].to_numpy()
    w = _fw_simplex(Y[:, :n_pre].T, y[:n_pre], zeta=1e-3, iters=2000, intercept=True)
    synth = w @ Y
    shift = (y[:n_pre] - synth[:n_pre]).mean()
    gap = y - synth - shift
    pre_rmspe = float(np.sqrt(np.mean(gap[:n_pre] ** 2)))
    post = gap[n_pre + PHASE_IN_WEEKS :]
    return {
        "weights": dict(zip(donors, w.round(4).tolist(), strict=True)),
        "gap": gap,
        "effect": float(post.mean()),
        "pre_rmspe": pre_rmspe,
        "post_rmspe": float(np.sqrt(np.mean(post**2))),
    }


def pooled(reps: int = 100) -> dict[str, Any]:
    """Every competitor opening in the history at once: openings fall on random dates, so the
    parallel-trends imputation estimator is credible here. Phase-in weeks are left out."""
    con = connect()
    df = con.execute(
        """
        select w.site_id, w.week, ln(w.sessions) as y, ln(greatest(w.availability, 0.05)) as lavail, s.segment, e.opened
        from site_week w join sites s using (site_id) left join competitor_events e using (site_id)
        where w.week >= date '2023-06-01' and s.opened < date '2023-01-02'
        """
    ).df()
    con.close()
    df["week"] = pd.to_datetime(df.week)
    df["opened"] = pd.to_datetime(df.opened)
    since = (df.week - df.opened).dt.days
    df = df[~((since >= 0) & (since < 7 * PHASE_IN_WEEKS))]
    df["treat"] = (since >= 7 * PHASE_IN_WEEKS).astype(int)
    p = impute.to_panel(df, "y", "segment", "lavail")

    def est(rows: np.ndarray | None) -> float:
        rows = np.arange(len(p.sites)) if rows is None else rows
        y0, usable = impute.fit_untreated(p, rows, trend=False)
        m = p.treated[rows] & ~np.isnan(p.y[rows]) & usable[:, None]
        return float(np.mean((p.y[rows] - y0)[m]))

    rng = np.random.default_rng(13)
    n = len(p.sites)
    draws = np.array([est(rng.integers(0, n, n)) for _ in range(reps)])
    sites = sorted(df.loc[df.treat == 1, "site_id"].unique())
    return {
        "estimate_log": est(None),
        "ci_log": np.percentile(draws, [2.5, 97.5]).tolist(),
        "sites": sites,
        "events": len(sites),
    }


def run() -> dict[str, Any]:
    cfg = analysis()["synthetic_control"]
    c = case()
    if c is None:
        return {"question": "What does a competitor's hub next to our site cost?", "available": False}
    wide, donors = _series(c["site_id"], c["opened"], cfg["pre_weeks"])
    weeks = list(wide.columns)
    n_pre = sum(1 for w in weeks if w < c["opened"])
    main = _fit(wide, c["site_id"], donors, n_pre)
    ratio = main["post_rmspe"] / max(main["pre_rmspe"], 1e-9)
    placebo = []
    for d in donors[: cfg["placebos"]]:
        others = [x for x in donors if x != d]
        f = _fit(wide, d, others, n_pre)
        placebo.append({"site_id": d, "effect": f["effect"], "ratio": f["post_rmspe"] / max(f["pre_rmspe"], 1e-9)})
    ratios = np.array([p["ratio"] for p in placebo])
    p_value = float((1 + (ratios >= ratio).sum()) / (1 + len(ratios)))
    effects = np.array([p["effect"] for p in placebo])
    # a placebo-based interval: the spread of effects where nothing happened
    lo, hi = main["effect"] + np.percentile(-effects, [2.5, 97.5])
    top = sorted(main["weights"].items(), key=lambda kv: -kv[1])[:6]
    return {
        "question": "What does a competitor's hub next to our site cost?",
        "available": True,
        "site_id": c["site_id"],
        "competitor_opened": str(c["opened"].date()),
        "donors": len(donors),
        "top_weights": [{"site_id": s, "weight": w} for s, w in top if w > 0],
        "pre_rmspe": main["pre_rmspe"],
        "effect_log": main["effect"],
        "effect_pct": float(np.expm1(main["effect"])),
        "ci_log": [float(lo), float(hi)],
        "placebo_p": p_value,
        "series": [
            {"week": str(pd.Timestamp(w).date()), "gap": float(g)} for w, g in zip(weeks, main["gap"], strict=True)
        ],
        "placebo_effects": effects.round(4).tolist(),
        "pooled": pooled(),
    }
