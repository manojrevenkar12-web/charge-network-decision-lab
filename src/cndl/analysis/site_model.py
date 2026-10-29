"""How busy would a new high-power site be? A Bayesian hierarchical model of mature demand.

Trained on the network's own established DC and HPC sites: mature demand per point (on the
capacity-linearised scale) explained by traffic, local EV density, segment, charger type, price,
competition from nearby sites, a competitor opening, and a country effect drawn from a common
distribution (countries with few sites borrow strength from the rest). Fitted with NUTS in
NumPyro. The posterior predictive for each candidate includes the site-to-site variation the
features cannot explain, which is most of the uncertainty a rollout faces.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
import pandas as pd
from numpyro.infer import MCMC, NUTS, Predictive

from ..settings import analysis
from ..warehouse import connect
from .economics import to_z

CAP = 36.0  # sessions a high-power point can serve per day (domain knowledge, as in question 1)
SEGMENTS = ["motorway", "urban_hub", "retail"]


def haversine(lat1: np.ndarray, lon1: np.ndarray, lat2: np.ndarray, lon2: np.ndarray) -> np.ndarray:
    p1, p2 = np.radians(lat1), np.radians(lat2)
    a = np.sin((p2 - p1) / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(np.radians(lon2 - lon1) / 2) ** 2
    return 6371.0 * 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def nearby(
    lat: np.ndarray, lon: np.ndarray, olat: np.ndarray, olon: np.ndarray, scale_km: float, exclude_self: bool
) -> np.ndarray:
    d = haversine(lat[:, None], lon[:, None], olat[None, :], olon[None, :])
    w = np.exp(-d / scale_km) * (d < 3 * scale_km)
    if exclude_self:
        np.fill_diagonal(w, 0.0)
    return w.sum(axis=1)


def training_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    cfg = analysis()["site_model"]
    con = connect()
    sites = con.execute("select * from sites").df()
    level = con.execute(
        """
        select d.site_id, sum(d.sessions) / count(*) as sessions_day, avg(d.availability) as availability,
               bool_or(p.site_id is not null) as raised
        from site_day d left join price_increase p using (site_id)
        where d.date >= date '2026-04-01' group by d.site_id
        """
    ).df()
    comp = con.execute("select site_id, max(opened) as comp_opened from competitor_events group by site_id").df()
    con.close()
    km = analysis()["rollout"]["cannibalisation_km"]
    sites["pressure"] = nearby(
        sites.lat.to_numpy(), sites.lon.to_numpy(), sites.lat.to_numpy(), sites.lon.to_numpy(), km, True
    )
    d = sites.merge(level, on="site_id").merge(comp, on="site_id", how="left")
    d["opened"] = pd.to_datetime(d.opened)
    mature = d.opened < pd.Timestamp("2026-09-28") - pd.Timedelta(days=cfg["mature_after_days"])
    d = d[mature & d.charger_type.isin(["DC", "HPC"]) & d.segment.isin(SEGMENTS)].copy()
    d["spd"] = d.sessions_day / d.availability.clip(lower=0.5) / d.points
    d["z"] = to_z(d.spd.to_numpy(), CAP)
    d["competitor"] = (pd.to_datetime(d.comp_opened) < pd.Timestamp("2026-02-01")).astype(float)
    return d, sites


# Price is deliberately not a feature: prices were raised where demand grew (question 1), so in
# a cross-section price "predicts" more demand. Its effect comes from question 1 instead.
FEATURES = ["log_traffic", "log_bev", "urban_hub", "retail", "hpc", "pressure", "competitor"]


def design(d: pd.DataFrame, countries: list[str]) -> tuple[np.ndarray, np.ndarray]:
    X = np.column_stack(
        [
            np.log(d.traffic_index),
            np.log(d.bev_density_index),
            (d.segment == "urban_hub").astype(float),
            (d.segment == "retail").astype(float),
            (d.charger_type == "HPC").astype(float) if "charger_type" in d else np.ones(len(d)),
            d.pressure,
            d.competitor if "competitor" in d else np.zeros(len(d)),
        ]
    )
    c = np.array([countries.index(x) for x in d.country])
    return X, c


def model(X: jnp.ndarray, c: jnp.ndarray, n_countries: int, y: jnp.ndarray | None = None) -> None:
    mu_a = numpyro.sample("mu_a", dist.Normal(2.0, 2.0))
    tau = numpyro.sample("tau", dist.HalfNormal(0.5))
    a_raw = numpyro.sample("a_raw", dist.Normal(0, 1).expand([n_countries]))
    a = numpyro.deterministic("a", mu_a + tau * a_raw)
    beta = numpyro.sample("beta", dist.Normal(0, 1).expand([X.shape[1]]))
    sigma = numpyro.sample("sigma", dist.HalfNormal(1.0))
    numpyro.sample("z", dist.Normal(a[c] + X @ beta, sigma), obs=y)


def fit(d: pd.DataFrame, countries: list[str], seed: int = 0) -> MCMC:
    cfg = analysis()["site_model"]
    X, c = design(d, countries)
    mcmc = MCMC(
        NUTS(model),
        num_warmup=cfg["warmup"],
        num_samples=cfg["samples"],
        num_chains=cfg["chains"],
        chain_method="sequential",
        progress_bar=False,
    )
    mcmc.run(jax.random.PRNGKey(seed), jnp.asarray(X), jnp.asarray(c), len(countries), jnp.asarray(d.z.to_numpy()))
    return mcmc


def predict(mcmc: MCMC, cand: pd.DataFrame, countries: list[str], draws: int, seed: int = 1) -> np.ndarray:
    """Posterior predictive draws of z for each candidate (draws x candidates)."""
    X, c = design(cand, countries)
    samples = mcmc.get_samples()
    idx = np.random.default_rng(seed).choice(len(samples["sigma"]), size=draws, replace=False)
    sub = {k: v[idx] for k, v in samples.items()}
    pred = Predictive(model, posterior_samples=sub, return_sites=["z"])
    return np.asarray(pred(jax.random.PRNGKey(seed), jnp.asarray(X), jnp.asarray(c), len(countries))["z"])


def diagnostics(mcmc: MCMC) -> dict[str, Any]:
    from numpyro.diagnostics import summary

    s = summary(mcmc.get_samples(group_by_chain=True))
    rhat = max(float(np.max(v["r_hat"])) for v in s.values())
    ess = min(float(np.min(v["n_eff"])) for v in s.values())
    beta = s["beta"]
    return {
        "max_r_hat": rhat,
        "min_ess": ess,
        "coefficients": {
            f: {"mean": float(beta["mean"][i]), "lo": float(beta["5.0%"][i]), "hi": float(beta["95.0%"][i])}
            for i, f in enumerate(FEATURES)
        },
        "sigma": float(s["sigma"]["mean"]),
        "tau": float(s["tau"]["mean"]),
    }
