"""Imputation estimator for staggered adoption with site-specific linear trends.

The untreated outcome of each site is modelled as its own intercept and linear trend plus a
week effect shared within its segment, fitted only on untreated weeks (never-treated sites and
treated sites before their first treated week), in the spirit of Borusyak, Jaravel and Spiess
(2024) with unit trends. Treated outcomes minus their imputed untreated outcomes are the effects.
Sites with too few untreated weeks to estimate a trend are left out and reported.

Fitting is by alternating projections on a dense sites-by-weeks matrix, which is fast enough to
bootstrap over sites.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class Panel:
    y: np.ndarray  # sites x weeks, NaN where missing
    treated: np.ndarray  # bool, sites x weeks
    group: np.ndarray  # int segment per site
    x: np.ndarray | None  # optional covariate, sites x weeks (log availability)
    sites: list[str]
    weeks: list[pd.Timestamp]


def to_panel(df: pd.DataFrame, y: str, group: str, x: str | None = None) -> Panel:
    sites = sorted(df.site_id.unique())
    weeks = sorted(df.week.unique())
    si = {s: i for i, s in enumerate(sites)}
    wi = {w: i for i, w in enumerate(weeks)}
    r = df.site_id.map(si).to_numpy()
    c = df.week.map(wi).to_numpy()
    shape = (len(sites), len(weeks))
    Y = np.full(shape, np.nan)
    Y[r, c] = df[y].to_numpy()
    T = np.zeros(shape, dtype=bool)
    T[r, c] = df["treat"].to_numpy().astype(bool)
    X = None
    if x:
        X = np.zeros(shape)
        X[r, c] = df[x].to_numpy()
    groups = df.drop_duplicates("site_id").set_index("site_id")[group].reindex(sites)
    codes = pd.Categorical(groups).codes
    return Panel(Y, T, codes, X, sites, list(weeks))


def fit_untreated(
    p: Panel, rows: np.ndarray | None = None, iters: int = 60, min_pre: int = 12, trend: bool = True
) -> tuple[np.ndarray, np.ndarray]:
    """Returns (imputed untreated outcome, usable site mask) for the given rows (with repeats).

    With trend=False this is the parallel-trends imputation estimator (site and week effects)."""
    rows = np.arange(len(p.sites)) if rows is None else rows
    Y = p.y[rows]
    T = p.treated[rows]
    G = p.group[rows]
    X = p.x[rows] if p.x is not None else None
    fit_mask = ~np.isnan(Y) & ~T
    usable = fit_mask.sum(axis=1) >= min_pre
    n, w = Y.shape
    t = np.arange(w, dtype=float)[None, :].repeat(n, axis=0)
    a = np.zeros(n)
    b = np.zeros(n)
    lam = np.zeros((G.max() + 1, w))
    beta = 0.0
    Yz = np.where(fit_mask, Y, 0.0)
    for _ in range(iters):
        # week effects by group
        resid = Yz - a[:, None] - b[:, None] * t - (beta * X if X is not None else 0.0)
        for g in range(lam.shape[0]):
            m = (g == G)[:, None] & fit_mask & usable[:, None]
            num = np.where(m, resid, 0.0).sum(axis=0)
            den = m.sum(axis=0)
            lam[g] = np.divide(num, den, out=np.zeros(w), where=den > 0)
        lam -= lam.mean(axis=1, keepdims=True)
        # site intercept and slope by least squares on its untreated weeks
        r2 = Yz - lam[G] - (beta * X if X is not None else 0.0)
        wts = fit_mask.astype(float)
        sw = wts.sum(1)
        st = (wts * t).sum(1)
        stt = (wts * t * t).sum(1)
        sy = (wts * r2).sum(1)
        sty = (wts * t * r2).sum(1)
        det = sw * stt - st * st
        ok = usable & (det > 1e-9)
        if trend:
            b = np.where(ok, (sw * sty - st * sy) / np.where(ok, det, 1), 0.0)
        a = np.where(ok, (sy - b * st) / np.where(sw > 0, sw, 1), 0.0)
        if X is not None:
            r3 = Yz - a[:, None] - b[:, None] * t - lam[G]
            xm = np.where(fit_mask & usable[:, None], X, 0.0)
            denx = (xm * xm).sum()
            beta = float((xm * r3).sum() / denx) if denx > 0 else 0.0
    y0 = a[:, None] + b[:, None] * t + lam[G] + (beta * X if X is not None else 0.0)
    return y0, usable


def att(p: Panel, rows: np.ndarray | None = None) -> tuple[float, int]:
    rows = np.arange(len(p.sites)) if rows is None else rows
    y0, usable = fit_untreated(p, rows)
    Y = p.y[rows]
    m = p.treated[rows] & ~np.isnan(Y) & usable[:, None]
    return float(np.mean((Y - y0)[m])), int(m.sum())


def bootstrap(p: Panel, reps: int, seed: int = 7) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = len(p.sites)
    return np.array([att(p, rng.integers(0, n, n))[0] for _ in range(reps)])


def effects_by_site(p: Panel) -> pd.DataFrame:
    y0, usable = fit_untreated(p)
    m = p.treated & ~np.isnan(p.y) & usable[:, None]
    r, c = np.nonzero(m)
    return pd.DataFrame({"site_id": np.array(p.sites)[r], "week": np.array(p.weeks)[c], "effect": (p.y - y0)[m]})


def event_time_effects(
    p: Panel, first: dict[str, int], leads: int, lags: int, rows: np.ndarray | None = None, trend: bool = False
) -> pd.DataFrame:
    """Placebo effects before treatment (fitting without the last `leads` pre weeks) and effects after."""
    rows_idx = np.arange(len(p.sites)) if rows is None else rows
    hidden = p.treated.copy()
    for i, s in enumerate(p.sites):
        g = first.get(s)
        if g is not None:
            hidden[i, max(0, g - leads) : g] = True
    q = Panel(p.y, hidden, p.group, p.x, p.sites, p.weeks)
    y0, usable = fit_untreated(q, rows_idx, trend=trend)
    out = []
    for r, i in enumerate(rows_idx):
        g = first.get(p.sites[i])
        if g is None or not usable[r]:
            continue
        for k in range(-leads, lags + 1):
            c = g + k
            if 0 <= c < p.y.shape[1] and not np.isnan(p.y[i, c]):
                out.append((k, p.y[i, c] - y0[r, c]))
    return pd.DataFrame(out, columns=["event_week", "effect"]).groupby("event_week").effect.mean().reset_index()
