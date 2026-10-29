"""Question 5: which sites, and how many points each, for next year's 1,000 high-power points?

1. Demand for every candidate comes from the hierarchical site model's posterior predictive
   (scenarios), shifted by the price effect found in question 1 (new sites are planned at the
   raised price) and grown to 2027 with question 4's country growth.
2. Each scenario, candidate and size (4, 8 or 12 points) gets a ten-year NPV from the shared
   economics: capex with a battery where the grid connection is too small, ramp-up, margins
   after energy, fees, partner share and opex.
3. A mixed-integer programme (HiGHS) picks sites and sizes to maximise expected NPV with about
   1,000 points, within the capex budget and the Italian motorway commitment, charging for
   new sites that would take each other's customers, and keeping the worst 10% of scenarios
   (CVaR) from losing money.

Three simpler plans are built the same way for comparison: highest traffic first, expected NPV
greedy without cannibalisation, and the programme without the risk constraint.
"""

from __future__ import annotations

from typing import Any

import highspy
import numpy as np
import pandas as pd

from ..settings import analysis
from ..warehouse import connect
from . import site_model as sm
from .economics import Costs, capex, npv


def candidates_frame() -> pd.DataFrame:
    con = connect()
    cand = con.execute("select * from candidates order by candidate_id").df()
    sites = con.execute("select lat, lon from sites").df()
    con.close()
    km = analysis()["rollout"]["cannibalisation_km"]
    cand["pressure"] = sm.nearby(
        cand.lat.to_numpy(), cand.lon.to_numpy(), sites.lat.to_numpy(), sites.lon.to_numpy(), km, False
    )
    cand["charger_type"] = "HPC"
    cand["competitor"] = 0.0
    return cand


def scenarios(
    cand: pd.DataFrame, price_shift_log: float, growth: dict[str, float], draws: int
) -> tuple[np.ndarray, dict[str, Any]]:
    d, sites = sm.training_data()
    countries = sorted(sites.country.unique())
    mcmc = sm.fit(d, countries)
    z = sm.predict(mcmc, cand, countries, draws)
    # the model learned demand at today's mix of raised and unraised prices; new sites are raised
    share_raised = float(d.raised.mean())
    z = z + price_shift_log * (1 - share_raised)
    # grown from the training window (mid-2026) to the first mature year (mid-2027)
    z = z + np.log(cand.country.map(growth).to_numpy())[None, :]
    diag = sm.diagnostics(mcmc)
    diag["training_sites"] = len(d)
    diag["share_raised"] = share_raised
    diag["pressure_coef"] = diag["coefficients"]["pressure"]["mean"]
    return z, diag


def _costs() -> Costs:
    return Costs.from_config(analysis()["rollout"], capacity_per_point=sm.CAP)


def value_table(cand: pd.DataFrame, z: np.ndarray, growth_yearly: np.ndarray) -> np.ndarray:
    """NPV for scenario x candidate x size."""
    c = _costs()
    options = np.array(analysis()["rollout"]["points_options"], dtype=float)
    return npv(
        z[:, :, None],
        options[None, None, :],
        cand.planned_price_eur_kwh.to_numpy()[None, :, None],
        cand.grid_kw.to_numpy()[None, :, None],
        growth_yearly[None, :, None],
        c,
    )


def _pairs(cand: pd.DataFrame, keep: np.ndarray, coef: float, margin_pv: np.ndarray) -> list[tuple[int, int, float]]:
    """Pairs of kept candidates close enough to share customers, with the NPV both would lose."""
    km = analysis()["rollout"]["cannibalisation_km"]
    idx = np.flatnonzero(keep)
    lat, lon = cand.lat.to_numpy()[idx], cand.lon.to_numpy()[idx]
    d = sm.haversine(lat[:, None], lon[:, None], lat[None, :], lon[None, :])
    out = []
    for a in range(len(idx)):
        for b in range(a + 1, len(idx)):
            if d[a, b] < 3 * km:
                w = np.exp(-d[a, b] / km)
                loss = (1 - np.exp(coef * w)) * (margin_pv[idx[a]] + margin_pv[idx[b]])
                out.append((int(idx[a]), int(idx[b]), float(loss)))
    return out


def optimise(
    cand: pd.DataFrame, V: np.ndarray, pairs: list[tuple[int, int, float]], risk: bool, keep: np.ndarray
) -> dict[str, Any]:
    cfg = analysis()["rollout"]
    options = np.array(cfg["points_options"], dtype=float)
    S, _, K = V.shape
    idx = np.flatnonzero(keep)
    n = len(idx)
    cap_k = capex(options[None, :], cand.grid_kw.to_numpy()[idx][:, None], _costs())  # n x K
    mean_v = V[:, idx, :].mean(axis=0)  # n x K
    h = highspy.Highs()
    h.setOptionValue("output_flag", False)
    h.setOptionValue("time_limit", 45.0)
    h.setOptionValue("mip_rel_gap", 0.005)
    inf = highspy.kHighsInf
    # columns: x (n*K binaries), y (pairs, continuous [0,1]), then eta and u (S) if risk
    nx, ny = n * K, len(pairs)
    obj = list(mean_v.ravel()) + [-p[2] for p in pairs]
    lb = [0.0] * (nx + ny)
    ub = [1.0] * (nx + ny)
    if risk:
        obj += [0.0] + [0.0] * S
        lb += [-inf] + [0.0] * S
        ub += [inf] + [inf] * S
    ncol = len(obj)
    h.addCols(
        ncol,
        np.array(obj),
        np.array(lb),
        np.array(ub),
        0,
        np.array([], dtype=np.int32),
        np.array([], dtype=np.int32),
        np.array([]),
    )
    h.changeColsIntegrality(nx, np.arange(nx, dtype=np.int32), np.array([highspy.HighsVarType.kInteger] * nx))
    h.changeObjectiveSense(highspy.ObjSense.kMaximize)
    rows: list[tuple[float, float, list[int], list[float]]] = []
    for i in range(n):  # one size per site
        rows.append((-inf, 1.0, [i * K + k for k in range(K)], [1.0] * K))
    pts = [float(options[k]) for _ in range(n) for k in range(K)]
    rows.append(
        (
            cfg["target_points"] - cfg["tolerance_points"],
            cfg["target_points"] + cfg["tolerance_points"],
            list(range(nx)),
            pts,
        )
    )
    rows.append((-inf, cfg["budget_eur"], list(range(nx)), list(cap_k.ravel())))
    for country, m in cfg["country_min_sites"].items():
        cols = [i * K + k for i in range(n) if cand.country.iloc[idx[i]] == country for k in range(K)]
        if cols:
            rows.append((m, inf, cols, [1.0] * len(cols)))
    for country in cand.country.iloc[idx].unique():
        cols = [i * K + k for i in range(n) if cand.country.iloc[idx[i]] == country for k in range(K)]
        vals = [float(options[k]) for i in range(n) if cand.country.iloc[idx[i]] == country for k in range(K)]
        rows.append((-inf, cfg["country_max_share"] * cfg["target_points"], cols, vals))
    pos = {c: j for j, c in enumerate(idx)}
    for p, (a, b, _) in enumerate(pairs):  # y >= x_a + x_b - 1
        ia, ib = pos[a], pos[b]
        cols = [ia * K + k for k in range(K)] + [ib * K + k for k in range(K)] + [nx + p]
        rows.append((-inf, 1.0, cols, [1.0] * (2 * K) + [-1.0]))
    if risk:
        eta = nx + ny
        alpha = cfg["cvar_alpha"]
        # eta - sum(u)/(alpha S) >= min_cvar
        rows.append(
            (cfg["min_cvar_eur"], inf, [eta] + [eta + 1 + s for s in range(S)], [1.0] + [-1.0 / (alpha * S)] * S)
        )
        for s in range(S):  # u_s >= eta - G_s  <=>  u_s - eta + G_s >= 0
            vs = V[s, idx, :].ravel()
            cols = list(range(nx)) + [nx + p for p in range(ny)] + [eta, eta + 1 + s]
            vals = list(vs) + [-p[2] for p in pairs] + [-1.0, 1.0]
            rows.append((0.0, inf, cols, vals))
    for lo, hi, cols, vals in rows:
        h.addRow(lo, hi, len(cols), np.array(cols, dtype=np.int32), np.array(vals))
    h.run()
    sol = np.array(h.getSolution().col_value)
    x = sol[:nx].reshape(n, K) > 0.5
    chosen = [(int(idx[i]), int(options[k])) for i in range(n) for k in range(K) if x[i, k]]
    return {
        "chosen": chosen,
        "status": h.modelStatusToString(h.getModelStatus()),
        "objective": float(h.getObjectiveValue()),
    }


def _plan_stats(
    cand: pd.DataFrame, V: np.ndarray, chosen: list[tuple[int, int]], pairs: list[tuple[int, int, float]]
) -> dict[str, Any]:
    options = analysis()["rollout"]["points_options"]
    sel = {i for i, _ in chosen}
    pen = sum(p[2] for p in pairs if p[0] in sel and p[1] in sel)
    g = np.zeros(V.shape[0])
    for i, k in chosen:
        g += V[:, i, options.index(k)]
    g -= pen
    alpha = analysis()["rollout"]["cvar_alpha"]
    tail = np.sort(g)[: max(1, int(alpha * len(g)))]
    caps = capex(
        np.array([k for _, k in chosen], dtype=float), cand.grid_kw.to_numpy()[[i for i, _ in chosen]], _costs()
    )
    return {
        "sites": len(chosen),
        "points": int(sum(k for _, k in chosen)),
        "capex_eur": float(caps.sum()),
        "expected_npv_eur": float(g.mean()),
        "cvar_npv_eur": float(tail.mean()),
        "p_loss": float((g < 0).mean()),
        "by_country": pd.Series([cand.country.iloc[i] for i, _ in chosen]).value_counts().to_dict(),
        "selection": [{"candidate_id": cand.candidate_id.iloc[i], "points": k} for i, k in chosen],
    }


def greedy(cand: pd.DataFrame, score: np.ndarray, points: int, target: int) -> list[tuple[int, int]]:
    out, total = [], 0
    for i in np.argsort(-score):
        if total + points > target:
            break
        out.append((int(i), points))
        total += points
    return out


def run(price_shift_log: float, growth: dict[str, float]) -> dict[str, Any]:
    cfg = analysis()["rollout"]
    cand = candidates_frame()
    z, diag = scenarios(cand, price_shift_log, growth, cfg["scenarios"])
    growth_yearly = cand.country.map(growth).to_numpy() - 1
    V = value_table(cand, z, growth_yearly)
    mean_v = V.mean(axis=0)
    # margin PV at 8 points without capex, for the cannibalisation charge
    k8 = cfg["points_options"].index(8)
    margin_pv = mean_v[:, k8] + capex(np.full(len(cand), 8.0), cand.grid_kw.to_numpy(), _costs())
    keep = mean_v.max(axis=1) > 0
    pairs = _pairs(cand, keep, diag["pressure_coef"], margin_pv)
    plans = {
        "highest_traffic": greedy(cand, cand.traffic_index.to_numpy(), 8, cfg["target_points"]),
        "expected_npv_greedy": greedy(cand, mean_v[:, k8], 8, cfg["target_points"]),
    }
    neutral = optimise(cand, V, pairs, risk=False, keep=keep)
    plans["optimised"] = neutral["chosen"]
    robust = optimise(cand, V, pairs, risk=True, keep=keep)
    plans["optimised_risk"] = robust["chosen"]
    stats = {k: _plan_stats(cand, V, v, pairs) for k, v in plans.items()}
    return {
        "question": "Which sites, and how many points each, for next year's 1,000 high-power points?",
        "candidates": len(cand),
        "candidates_with_positive_npv": int(keep.sum()),
        "close_pairs": len(pairs),
        "model": diag,
        "assumptions": {"price_shift_log": price_shift_log, "growth": growth},
        "solver": {"optimised": neutral["status"], "optimised_risk": robust["status"]},
        "plans": stats,
        "recommended": "optimised_risk",
        "candidate_npv": [
            {
                "candidate_id": cand.candidate_id.iloc[i],
                "country": cand.country.iloc[i],
                "lat": float(cand.lat.iloc[i]),
                "lon": float(cand.lon.iloc[i]),
                "segment": cand.segment.iloc[i],
                "npv_mean": float(mean_v[i, k8]),
                "npv_p10": float(np.percentile(V[:, i, k8], 10)),
                "npv_p90": float(np.percentile(V[:, i, k8], 90)),
            }
            for i in range(len(cand))
        ],
    }
