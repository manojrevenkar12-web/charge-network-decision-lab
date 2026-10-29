"""The simulated charge-point network and its true structural model.

Everything the analyses see is drawn from here: a site table without the hidden quantities,
daily sessions, energy and revenue, the hourly data of the off-peak trial, competitor openings
and the candidate locations for next year's rollout. The quantities an analysis is trying to
estimate (price elasticities, the effect of the 2025 price increase, a competitor's effect, the
trial's effects, next year's demand, each candidate's mature demand) are written to the truth
directory, which only `cndl.evaluation` reads.

The world is built so that the obvious analysis is wrong in a realistic way: country managers
raised prices in 2025 where demand was already growing faster, so a before-and-after or two-way
fixed effects comparison understates how much the price increase cost in sessions.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

# rough land boxes (lat, lon) for coordinates; distances only matter within a country
BOXES = {
    "DE": (47.5, 54.8, 6.0, 14.8),
    "NL": (51.0, 53.4, 3.6, 7.1),
    "IT": (38.0, 46.4, 7.5, 16.5),
    "GB": (50.3, 55.8, -4.5, 1.6),
    "SE": (55.4, 60.5, 11.5, 18.5),
    "DK": (54.8, 57.6, 8.2, 12.6),
    "AT": (46.5, 48.9, 9.6, 16.9),
    "PL": (49.2, 54.7, 14.2, 23.9),
    "CZ": (48.6, 51.0, 12.2, 18.8),
    "HU": (45.8, 48.5, 16.2, 22.8),
    "RO": (43.7, 48.2, 20.3, 29.6),
}


def _cities(rng: np.random.Generator, per_country: int = 6) -> dict[str, np.ndarray]:
    out = {}
    for c, b in BOXES.items():
        out[c] = np.column_stack(
            [rng.uniform(b[0] + 0.3, b[1] - 0.3, per_country), rng.uniform(b[2] + 0.3, b[3] - 0.3, per_country)]
        )
    return out


def _place(rng: np.random.Generator, country: str, segment: str, cities: dict[str, np.ndarray]) -> tuple[float, float]:
    """Motorway sites anywhere in the country; the rest cluster around its cities."""
    b = BOXES[country]
    if segment == "motorway":
        return float(rng.uniform(b[0], b[1])), float(rng.uniform(b[2], b[3]))
    c = cities[country][rng.integers(0, len(cities[country]))]
    return float(c[0] + rng.normal(0, 0.10)), float(c[1] + rng.normal(0, 0.15))


def pressure(
    lat: np.ndarray, lon: np.ndarray, olat: np.ndarray, olon: np.ndarray, scale_km: float, exclude_self: bool
) -> np.ndarray:
    """Competition from nearby charging sites: sum of exp(-distance/scale) within three scales."""
    d = km(lat[:, None], lon[:, None], olat[None, :], olon[None, :])
    w = np.exp(-d / scale_km) * (d < 3 * scale_km)
    if exclude_self:
        np.fill_diagonal(w, 0.0)
    return w.sum(axis=1)


def load_config(path: str | Path = "config/world.yaml", overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    cfg = yaml.safe_load(Path(path).read_text())
    if overrides:
        cfg = _merge(cfg, overrides)
    return cfg


def _merge(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    out = dict(a)
    for k, v in b.items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def km(lat1: np.ndarray, lon1: np.ndarray, lat2: np.ndarray, lon2: np.ndarray) -> np.ndarray:
    """Great-circle distance in km (broadcasting)."""
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi, dl = p2 - p1, np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 6371.0 * 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


@dataclass
class World:
    observed: dict[str, pd.DataFrame]
    truth: dict[str, Any]
    config: dict[str, Any] | None = None


def _days(cfg: dict[str, Any]) -> pd.DatetimeIndex:
    start = pd.Timestamp(cfg["start"])
    end = pd.Timestamp(cfg["end"]) + pd.Timedelta(weeks=cfg["future_weeks"])
    return pd.date_range(start, end, freq="D")


def _sites(rng: np.random.Generator, cfg: dict[str, Any], cities: dict[str, np.ndarray]) -> pd.DataFrame:
    n = cfg["network"]["sites"]
    countries = list(cfg["countries"])
    shares = np.array([cfg["countries"][c]["share"] for c in countries])
    country = rng.choice(countries, size=n, p=shares / shares.sum())
    segs = cfg["network"]["segments"]
    seg_names = list(segs)
    seg_p = np.array([segs[s]["share"] for s in seg_names])
    segment = rng.choice(seg_names, size=n, p=seg_p / seg_p.sum())
    types = cfg["network"]["types"]
    ctype = np.array([rng.choice(list(segs[s]["types"]), p=np.array(list(segs[s]["types"].values()))) for s in segment])
    points = np.array([rng.integers(types[t]["points"][0], types[t]["points"][1] + 1) for t in ctype])
    points = np.where(ctype == "HPC", 2 * np.round(points / 2), points).astype(int)
    # most sites opened before the history starts; the network keeps growing during it
    start = pd.Timestamp(cfg["start"])
    age_days = np.where(rng.random(n) < 0.62, rng.integers(200, 2000, n), -rng.integers(0, 1250, n))
    opened = start - pd.to_timedelta(age_days, unit="D")
    lat = np.empty(n)
    lon = np.empty(n)
    for i, c in enumerate(country):
        lat[i], lon[i] = _place(rng, c, segment[i], cities)
    traffic = np.exp(rng.normal(0, 0.5, n) + np.where(segment == "motorway", 0.4, 0.0))
    bev_density = np.exp(rng.normal(0, 0.3, n))
    elasticity = np.array([segs[s]["elasticity"] for s in segment]) + rng.normal(
        0, cfg["network"]["elasticity_site_sd"], n
    )
    grid_kw = np.round(rng.lognormal(np.log(1600), 0.5, n), -1)
    return pd.DataFrame(
        {
            "site_id": [f"{c}-{i:04d}" for i, c in enumerate(country)],
            "country": country,
            "segment": segment,
            "charger_type": ctype,
            "points": points,
            "opened": opened.normalize(),
            "lat": lat.round(4),
            "lon": lon.round(4),
            "traffic_index": traffic.round(3),
            "bev_density_index": bev_density.round(3),
            "grid_kw": grid_kw,
            # hidden
            "quality": rng.normal(0, cfg["network"]["quality_sd"], n),
            "local_trend": rng.normal(0, cfg["network"]["local_trend_sd"], n),
            "elasticity": elasticity,
        }
    )


def _seasonal(cfg: dict[str, Any], segment: np.ndarray, days: pd.DatetimeIndex) -> np.ndarray:
    seas = cfg["network"]["seasonality"]
    doy = days.dayofyear.to_numpy()
    dow = days.dayofweek.to_numpy()
    out = np.zeros((len(segment), len(days)))
    for s, p in seas.items():
        m = segment == s
        annual = p["annual"] * np.cos(2 * np.pi * (doy - p["peak_day"]) / 365.25)
        weekly = np.log(np.array(p["weekly"]))[dow]
        out[m] = annual + weekly
    return out


def _smooth_log_price(logp: np.ndarray, halflife: float) -> np.ndarray:
    """Drivers adjust to a price change gradually: exponential smoothing along time."""
    a = 1 - np.exp(-np.log(2) / halflife)
    out = np.empty_like(logp)
    out[:, 0] = logp[:, 0]
    for t in range(1, logp.shape[1]):
        out[:, t] = out[:, t - 1] + a * (logp[:, t] - out[:, t - 1])
    return out


def build(cfg: dict[str, Any]) -> World:
    rng = np.random.default_rng(cfg["seed"])
    net = cfg["network"]
    cities = _cities(rng)
    sites = _sites(rng, cfg, cities)
    days = _days(cfg)
    n, d = len(sites), len(days)
    hist_end = pd.Timestamp(cfg["end"])
    in_hist = days <= hist_end
    t_years = ((days - pd.Timestamp(cfg["start"])).days.to_numpy() / 365.25)[None, :]

    seg = sites["segment"].to_numpy()
    ctype = sites["charger_type"].to_numpy()
    types = net["types"]
    per_point = np.array([types[t]["per_point"] for t in ctype])
    cap = np.array([types[t]["cap_per_point"] for t in ctype]) * sites["points"].to_numpy()
    kwh_per = np.array([types[t]["kwh"] for t in ctype])
    growth = np.array([cfg["countries"][c]["growth"] for c in sites["country"]])
    price_idx = np.array([cfg["countries"][c]["price"] for c in sites["country"]])
    base_price = np.round(np.array([types[t]["base_price"] for t in ctype]) * price_idx, 2)

    # --- structural log demand, without price, competitors and outages ---------------------
    age = (days.to_numpy()[None, :] - sites["opened"].to_numpy()[:, None]).astype("timedelta64[D]").astype(float)
    open_mask = age >= 0
    ramp = np.log(np.clip(1 - net["ramp_depth"] * np.exp(-np.clip(age, 0, None) / net["ramp_days"]), 1e-6, None))
    base = (
        np.log(sites["points"].to_numpy() * per_point)[:, None]
        + net["traffic_beta"] * np.log(sites["traffic_index"].to_numpy())[:, None]
        + 0.25 * np.log(sites["bev_density_index"].to_numpy())[:, None]
        + sites["quality"].to_numpy()[:, None]
        + net["bev_beta"] * np.log1p(growth)[:, None] * t_years
        + sites["local_trend"].to_numpy()[:, None] * t_years
        + _seasonal(cfg, seg, days)
        + ramp
        - cfg["rollout"]["cannibalisation"]
        * pressure(
            sites.lat.to_numpy(),
            sites.lon.to_numpy(),
            sites.lat.to_numpy(),
            sites.lon.to_numpy(),
            cfg["rollout"]["cannibalisation_km"],
            True,
        )[:, None]
    )
    # country shocks (weather, holidays): AR(1) by country, shared by its sites
    shocks = {}
    for c in cfg["countries"]:
        e = rng.normal(0, net["country_shock_sd"], d)
        s = np.zeros(d)
        for t in range(1, d):
            s[t] = net["country_shock_ar"] * s[t - 1] + e[t]
        shocks[c] = s
    base += np.vstack([shocks[c] for c in sites["country"]])

    # --- competitors ------------------------------------------------------------------------
    comp = cfg["competitors"]
    comp_effect = np.zeros((n, d))
    comp_sites = rng.random(n) < comp["share_of_sites"]
    w0, w1 = pd.Timestamp(comp["window"][0]), pd.Timestamp(comp["window"][1])
    events = []
    true_comp = {}
    for i in np.flatnonzero(comp_sites):
        when = w0 + pd.Timedelta(days=int(rng.integers(0, (w1 - w0).days)))
        eff = (
            comp["effect_mean"]
            + rng.normal(0, comp["effect_sd"])
            + (comp["urban_extra"] if seg[i] == "urban_hub" else 0.0)
        )
        k = np.clip((days - when).days.to_numpy() / comp["phase_in_days"], 0, 1)
        comp_effect[i] = eff * k
        events.append(
            {"site_id": sites.site_id[i], "opened": when.normalize(), "operator": f"competitor-{rng.integers(1, 7)}"}
        )
        true_comp[sites.site_id[i]] = {
            "structural": float(eff),
            "from": (when + pd.Timedelta(days=comp["phase_in_days"])).normalize(),
        }

    # --- prices: base, and the 2025 increase targeted at growing sites -----------------------
    pc = cfg["price_change"]
    util24 = sites["quality"].to_numpy() + 0.0  # managers see utilisation, which reflects quality
    z = lambda x: (x - x.mean()) / x.std()  # noqa: E731
    logit = (
        pc["select_intercept"]
        + pc["select_on_trend"] * z(sites["local_trend"].to_numpy())
        + pc["select_on_utilisation"] * z(util24)
    )
    opened_before = sites["opened"] < pd.Timestamp(pc["waves"][0]) - pd.Timedelta(days=180)
    selected = (rng.random(n) < 1 / (1 + np.exp(-logit))) & opened_before.to_numpy()
    wave = np.where(selected, rng.integers(0, len(pc["waves"]), n), -1)
    price1 = np.repeat(base_price[:, None], d, axis=1)
    treat_start = np.full(n, np.datetime64("NaT"), dtype="datetime64[ns]")
    for i in np.flatnonzero(selected):
        ws = pd.Timestamp(pc["waves"][wave[i]])
        price1[i, days >= ws] += pc["delta_eur"]
        treat_start[i] = np.datetime64(ws)
    elast = sites["elasticity"].to_numpy()[:, None]
    lp1 = _smooth_log_price(np.log(price1 / base_price[:, None]), net["price_halflife_days"])
    lp0 = np.zeros_like(lp1)

    # --- outages ----------------------------------------------------------------------------
    out = cfg["outages"]
    down = np.zeros((n, d))
    starts = rng.random((n, d)) < out["daily_rate"]
    for i, t in zip(*np.nonzero(starts), strict=True):
        length = int(rng.geometric(1 / out["mean_days"]))
        share = rng.integers(1, sites.points[i] + 1) / sites.points[i]
        down[i, t : t + length] = np.maximum(down[i, t : t + length], share)
    avail = 1 - down

    def expected(lp: np.ndarray, competitors: bool = True) -> np.ndarray:
        log_mu = base + elast * lp + (comp_effect if competitors else 0.0)
        mu = np.exp(log_mu) * (1 - out["demand_loss"] * down)
        mu = cap[:, None] * (1 - np.exp(-mu / cap[:, None]))  # queues saturate a busy site
        return np.where(open_mask, mu, 0.0)

    mu1 = expected(lp1)
    mu0 = expected(lp0)
    mu_nc = expected(lp1, competitors=False)
    # the counterfactual without the competitor, only for the site the synthetic control studies
    size = net["dispersion"]
    sessions = rng.negative_binomial(size, size / (size + np.maximum(mu1, 1e-9)))
    sessions = np.where(open_mask, sessions, 0)
    kwh = sessions * kwh_per[:, None] * np.exp(rng.normal(0, 0.05, (n, d)))
    revenue = kwh * price1

    # --- the off-peak switchback trial (hourly) ------------------------------------------------
    tr = cfg["offpeak_trial"]
    t0 = pd.Timestamp(tr["start"])
    trial_days = pd.date_range(t0, periods=7 * tr["weeks"], freq="D")
    pre_days = pd.date_range(t0 - pd.Timedelta(weeks=tr["weeks"]), periods=7 * tr["weeks"], freq="D")
    eligible = np.flatnonzero(
        (ctype == "HPC")
        & np.isin(seg, ["motorway", "urban_hub"])
        & (sites["opened"] < t0 - pd.Timedelta(days=365)).to_numpy()
    )
    trial_sites = np.sort(rng.choice(eligible, size=min(tr["sites"], len(eligible)), replace=False))
    assign = rng.random((len(trial_sites), len(trial_days))) < 0.5
    off = np.zeros(24, dtype=bool)
    off[tr["hours"]] = True
    hourly_rows = []
    truth_trial = {"revenue_effect": [], "offpeak_share_effect": [], "peak_sessions_effect": []}
    for k, i in enumerate(trial_sites):
        shape = np.array(tr["hourly_shape"][seg[i]])
        shape = shape / shape.sum()
        for when, is_trial in [(pre_days, False), (trial_days, True)]:
            for j, day in enumerate(when):
                t = days.get_loc(day)
                mu_day = mu1[i, t]
                disc = bool(assign[k, j]) if is_trial else False
                prev = bool(assign[k, j - 1]) if (is_trial and j > 0) else False
                mult, mult0 = _trial_multipliers(off, tr, disc, prev)
                lam = mu_day * shape * mult
                s = rng.poisson(lam)
                e = s * kwh_per[i] * np.exp(rng.normal(0, 0.05, 24))
                p = np.where(off & disc, price1[i, t] * (1 - tr["discount"]), price1[i, t])
                for h in range(24):
                    hourly_rows.append(
                        (sites.site_id[i], day, h, int(s[h]), float(e[h]), float(p[h]), disc and is_trial)
                    )
                if is_trial:
                    sessions[i, t] = int(s.sum())
                    kwh[i, t] = float(e.sum())
                    revenue[i, t] = float((e * p).sum())
                    # truth: discount vs no discount on the same day, without carryover
                    lam_d = mu_day * shape * _trial_multipliers(off, tr, True, False)[0]
                    lam_n = mu_day * shape
                    r_d = (lam_d * kwh_per[i] * np.where(off, price1[i, t] * (1 - tr["discount"]), price1[i, t])).sum()
                    r_n = (lam_n * kwh_per[i] * price1[i, t]).sum()
                    truth_trial["revenue_effect"].append(r_d - r_n)
                    truth_trial["offpeak_share_effect"].append(
                        lam_d[off].sum() / lam_d.sum() - lam_n[off].sum() / lam_n.sum()
                    )
                    truth_trial["peak_sessions_effect"].append(lam_d[~off].sum() / lam_n[~off].sum() - 1)
                    _ = mult0
    hourly = pd.DataFrame(
        hourly_rows, columns=["site_id", "date", "hour", "sessions", "kwh", "price_eur_kwh", "discount_day"]
    )
    trial_assign = pd.DataFrame(
        [
            (sites.site_id[i], day, bool(assign[k, j]))
            for k, i in enumerate(trial_sites)
            for j, day in enumerate(trial_days)
        ],
        columns=["site_id", "date", "discount"],
    )

    # --- observed daily panel (history only) ---------------------------------------------------
    hist = np.flatnonzero(in_hist)
    rows = np.nonzero(open_mask[:, hist])
    daily = pd.DataFrame(
        {
            "site_id": sites.site_id.to_numpy()[rows[0]],
            "date": days[hist][rows[1]],
            "sessions": sessions[:, hist][rows].astype(int),
            "kwh": np.round(kwh[:, hist][rows], 1),
            "revenue_eur": np.round(revenue[:, hist][rows], 2),
            "price_eur_kwh": np.round(price1[:, hist][rows], 2),
            "availability": np.round(avail[:, hist][rows], 3),
        }
    )

    # --- truth ----------------------------------------------------------------------------------
    weeks = days.to_period("W-SUN")
    treated = np.flatnonzero(selected)
    att_rows = []
    for i in treated:
        post = (days >= pd.Timestamp(treat_start[i])) & in_hist
        wk = pd.Series(mu1[i, post], index=weeks[post]).groupby(level=0).sum()
        wk0 = pd.Series(mu0[i, post], index=weeks[post]).groupby(level=0).sum()
        days_in = pd.Series(1, index=weeks[post]).groupby(level=0).sum()
        full = days_in.index[days_in.to_numpy() == 7]
        for p in full:
            att_rows.append((sites.site_id[i], seg[i], str(p), float(np.log(wk[p] / wk0[p])), float(wk[p] - wk0[p])))
    att = pd.DataFrame(att_rows, columns=["site_id", "segment", "week", "log_effect", "sessions_effect"])

    # future: expected and realised weekly sessions and energy, status quo
    fut = np.flatnonzero(~in_hist)
    fut_mu = mu1[:, fut]
    fut_sessions = rng.negative_binomial(size, size / (size + np.maximum(fut_mu, 1e-9)))
    future = pd.DataFrame(
        {
            "site_id": np.repeat(sites.site_id.to_numpy(), len(fut)),
            "country": np.repeat(sites.country.to_numpy(), len(fut)),
            "date": np.tile(days[fut], n),
            "expected_kwh": (fut_mu * kwh_per[:, None]).ravel(),
            "kwh": (fut_sessions * kwh_per[:, None]).ravel(),
        }
    )

    # competitor effects as realised: weekly expected sessions with and without the competitor
    comp_truth = {}
    for sid, v in true_comp.items():
        i = int(np.flatnonzero(sites.site_id == sid)[0])
        post = (days >= v["from"]) & in_hist
        if post.sum() < 7:
            continue
        wk1 = pd.Series(mu1[i, post], index=weeks[post]).groupby(level=0).sum()
        wk0 = pd.Series(mu_nc[i, post], index=weeks[post]).groupby(level=0).sum()
        comp_truth[sid] = {"structural": v["structural"], "log_effect": float(np.mean(np.log(wk1 / wk0)))}
    # the competitor case for the synthetic control: a large motorway site hit mid-history
    sc_site = _synthetic_control_case(sites, events, cfg)
    sc_truth = None
    if sc_site is not None and sc_site in comp_truth:
        when = next(e["opened"] for e in events if e["site_id"] == sc_site)
        sc_truth = {"site_id": sc_site, "opened": str(when.date()), "log_effect": comp_truth[sc_site]["log_effect"]}

    candidates, cand_truth = _candidates(rng, cfg, sites, cities)

    observed = {
        "sites": sites.drop(columns=["quality", "local_trend", "elasticity"]),
        "daily": daily,
        "prices": _price_changes(sites, price1, base_price, days, in_hist),
        "competitor_events": pd.DataFrame(events),
        "trial_hourly": hourly,
        "trial_assignment": trial_assign,
        "candidates": candidates,
    }
    truth = {
        "site_params": sites[["site_id", "quality", "local_trend", "elasticity"]],
        "price_change_att": att,
        "price_change_summary": {
            "att_log": float(att.log_effect.mean()),
            "att_pct": float(np.exp(att.log_effect.mean()) - 1),
            "by_segment_log": att.groupby("segment").log_effect.mean().to_dict(),
            "treated_sites": len(treated),
        },
        "competitor_effects": comp_truth,
        "synthetic_control": sc_truth,
        "trial": {k: float(np.mean(v)) for k, v in truth_trial.items()},
        "future": future,
        "candidates": cand_truth,
    }
    return World(observed=observed, truth=truth, config=cfg)


def _trial_multipliers(off: np.ndarray, tr: dict[str, Any], disc: bool, prev: bool) -> tuple[np.ndarray, np.ndarray]:
    mult = np.ones(24)
    if disc:
        mult[off] *= 1 + tr["offpeak_gain"]
        mult[~off] *= 1 - tr["peak_shift"]
    elif prev:
        mult[off] *= 1 + tr["carryover"] * tr["offpeak_gain"]
    return mult, np.ones(24)


def _price_changes(
    sites: pd.DataFrame, price: np.ndarray, base: np.ndarray, days: pd.DatetimeIndex, in_hist: np.ndarray
) -> pd.DataFrame:
    rows = []
    for i in range(len(sites)):
        rows.append((sites.site_id[i], days[0], float(base[i])))
        ch = np.flatnonzero(np.diff(price[i, in_hist]) != 0)
        for t in ch:
            rows.append((sites.site_id[i], days[t + 1], float(round(price[i, t + 1], 2))))
    return pd.DataFrame(rows, columns=["site_id", "valid_from", "price_eur_kwh"])


def _synthetic_control_case(sites: pd.DataFrame, events: list[dict[str, Any]], cfg: dict[str, Any]) -> str | None:
    lo, hi = pd.Timestamp("2025-01-01"), pd.Timestamp("2025-12-31")
    pool = [e for e in events if lo <= e["opened"] <= hi]
    big = [
        e["site_id"]
        for e in pool
        if (row := sites.loc[sites.site_id == e["site_id"]].iloc[0]).segment == "motorway"
        and row.charger_type == "HPC"
        and row.opened < pd.Timestamp(cfg["start"])
    ]
    return sorted(big)[0] if big else None


def _candidates(
    rng: np.random.Generator, cfg: dict[str, Any], sites: pd.DataFrame, cities: dict[str, np.ndarray]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    ro = cfg["rollout"]
    m = ro["candidates"]
    net = cfg["network"]
    countries = list(cfg["countries"])
    shares = np.array([cfg["countries"][c]["share"] for c in countries])
    country = rng.choice(countries, size=m, p=shares / shares.sum())
    segment = rng.choice(["motorway", "urban_hub", "retail"], size=m, p=[0.45, 0.35, 0.20])
    lat = np.empty(m)
    lon = np.empty(m)
    for i, c in enumerate(country):
        lat[i], lon[i] = _place(rng, c, segment[i], cities)
    traffic = np.exp(rng.normal(0, 0.5, m) + np.where(segment == "motorway", 0.4, 0.0))
    bev = np.exp(rng.normal(0, 0.3, m))
    quality = rng.normal(0, net["quality_sd"], m)
    grid_kw = np.round(rng.lognormal(np.log(1700), 0.6, m), -1)
    growth = np.array([cfg["countries"][c]["growth"] for c in country])
    # mature sessions per point per day in the first full year (2027), before cannibalisation
    years_from_start = (pd.Timestamp("2027-07-01") - pd.Timestamp(cfg["start"])).days / 365.25
    log_pp = (
        np.log(net["types"]["HPC"]["per_point"])
        + net["traffic_beta"] * np.log(traffic)
        + 0.25 * np.log(bev)
        + quality
        + net["bev_beta"] * np.log1p(growth) * years_from_start
    )
    # nearby existing sites take demand away
    press = pressure(lat, lon, sites["lat"].to_numpy(), sites["lon"].to_numpy(), ro["cannibalisation_km"], False)
    elast = np.array([net["segments"][s]["elasticity"] for s in segment])
    base_p = net["types"]["HPC"]["base_price"] * np.array([cfg["countries"][c]["price"] for c in country])
    price = np.round(base_p + cfg["price_change"]["delta_eur"], 2)
    log_pp_c = log_pp - ro["cannibalisation"] * press + elast * np.log(price / base_p)
    cap = net["types"]["HPC"]["cap_per_point"]
    cands = pd.DataFrame(
        {
            "candidate_id": [f"C-{c}-{i:04d}" for i, c in enumerate(country)],
            "country": country,
            "segment": segment,
            "lat": lat.round(4),
            "lon": lon.round(4),
            "traffic_index": traffic.round(3),
            "bev_density_index": bev.round(3),
            "grid_kw": grid_kw,
            "planned_price_eur_kwh": price,
            "partner": rng.choice(["fuel_retail", "supermarket", "motorway_operator", "municipality", "hotel"], size=m),
        }
    )
    truth = pd.DataFrame(
        {
            "candidate_id": cands.candidate_id,
            # before cannibalisation between new sites, which depends on what is built together
            "sessions_per_point_day": cap * (1 - np.exp(-np.exp(log_pp_c) / cap)),
            "log_demand": log_pp_c,
            "quality": quality,
            "elasticity": elast,
        }
    )
    return cands, truth


def write(world: World, out: Path, truth_dir: Path) -> dict[str, int]:
    out.mkdir(parents=True, exist_ok=True)
    truth_dir.mkdir(parents=True, exist_ok=True)
    counts = {}
    for name, df in world.observed.items():
        df.to_parquet(out / f"{name}.parquet", index=False)
        counts[name] = len(df)
    if world.config is not None:
        (truth_dir / "world_config.json").write_text(json.dumps(world.config, indent=2, default=str))
    for name, v in world.truth.items():
        if isinstance(v, pd.DataFrame):
            v.to_parquet(truth_dir / f"{name}.parquet", index=False)
        else:
            (truth_dir / f"{name}.json").write_text(json.dumps(v, indent=2, default=str))
    return counts


def today() -> date:
    return date(2026, 9, 28)


__all__ = ["World", "build", "km", "load_config", "timedelta", "today", "write"]
