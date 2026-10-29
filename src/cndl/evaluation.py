"""Judges every answer against the simulator's truth. The only module besides cndl.sim that reads
build/truth or config/world.yaml (tests/test_isolation.py enforces it).

For each question: is the estimate close, does its interval cover the truth, and is the decision
the one the truth supports. For the rollout: what each plan would really earn, against the best
plan that knowing the true demand allows (regret).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .analysis import economics, rollout
from .analysis import site_model as sm
from .settings import ROOT, TRUTH, analysis
from .sim.world import load_config


def _covers(ci: list[float], truth: float) -> bool:
    return bool(ci[0] <= truth <= ci[1])


def price_change(res: dict[str, Any], truth_dir: Path) -> dict[str, Any]:
    att = pd.read_parquet(truth_dir / "price_change_att.parquet")
    truth = float(att[att.site_id.isin(res["analysed_sites"])].log_effect.mean())
    methods = [*res["usual_estimates"], *res["credible_estimates"]]
    seen = set()
    rows = []
    for m in methods:
        if m["method"] in seen:
            continue
        seen.add(m["method"])
        rows.append(
            {
                "method": m["method"],
                "estimate": m["estimate"],
                "error": m["estimate"] - truth,
                "covers": _covers(m["ci"], truth),
            }
        )
    by_seg = att[att.site_id.isin(res["analysed_sites"])].groupby("segment").log_effect.mean().to_dict()
    return {
        "truth_log": truth,
        "methods": rows,
        "range_covers_truth": _covers(res["range_log"], truth),
        "range_width": res["range_log"][1] - res["range_log"][0],
        "truth_by_segment": by_seg,
        "decision_correct": (res["decision"] == "keep") == bool(np.expm1(truth + res["price_increase_log"]) > 0),
    }


def trial(res: dict[str, Any], truth_dir: Path) -> dict[str, Any]:
    t = json.loads((truth_dir / "trial.json").read_text())
    e = res["effects"]
    pairs = {
        "revenue_eur": t["revenue_effect"],
        "offpeak_share": t["offpeak_share_effect"],
        "log_peak_sessions": float(np.log1p(t["peak_sessions_effect"])),
    }
    out = {k: {"truth": v, "estimate": e[k]["estimate"], "covers": _covers(e[k]["ci"], v)} for k, v in pairs.items()}
    # the true net value per site-day, with the same value per relieved daytime session
    relieved = (
        -res["peak_sessions_relieved_per_site_day"][1]
        / np.expm1(e["log_peak_sessions"]["estimate"])
        * t["peak_sessions_effect"]
    )
    net = t["revenue_effect"] + res["peak_session_value_eur"] * relieved
    out["true_net_value_per_site_day"] = float(net)
    out["decision_correct"] = (
        (res["decision"] == "roll out") == (net > 0) if res["decision"] != "extend the trial" else None
    )
    return out


def competitor(res: dict[str, Any], truth_dir: Path) -> dict[str, Any]:
    if not res.get("available"):
        return {"available": False}
    sc = json.loads((truth_dir / "synthetic_control.json").read_text())
    comp = json.loads((truth_dir / "competitor_effects.json").read_text())
    pooled_truth = float(np.mean([comp[s]["log_effect"] for s in res["pooled"]["sites"] if s in comp]))
    return {
        "available": True,
        "synthetic_control": {
            "truth": sc["log_effect"] if sc else None,
            "estimate": res["effect_log"],
            "covers": _covers(res["ci_log"], sc["log_effect"]) if sc else None,
        },
        "pooled": {
            "truth": pooled_truth,
            "estimate": res["pooled"]["estimate_log"],
            "covers": _covers(res["pooled"]["ci_log"], pooled_truth),
        },
    }


def forecast(res: dict[str, Any], truth_dir: Path) -> dict[str, Any]:
    fut = pd.read_parquet(truth_dir / "future.parquet")
    fut["month"] = fut.date.dt.to_period("M").dt.start_time
    full = fut.groupby(["country", "month"]).agg(kwh=("kwh", "sum"), days=("date", "nunique")).reset_index()
    full = full[full.days == full.month.dt.days_in_month]
    f = pd.DataFrame(res["forecast"])
    f["month"] = pd.to_datetime(f.month)
    j = f.merge(full, on=["country", "month"], suffixes=("_fc", "_true"))
    tot = j.groupby("month")[["kwh_fc", "kwh_true"]].sum()
    t = pd.DataFrame(res["total"])
    t["month"] = pd.to_datetime(t.month)
    t = t.merge(tot.kwh_true.reset_index(), on="month")
    return {
        "months": int(j.month.nunique()),
        "mape_country": float(np.mean(np.abs(j.kwh_true / j.kwh_fc - 1))),
        "mape_total": float(np.mean(np.abs(tot.kwh_true / tot.kwh_fc - 1))),
        "bias_total": float(tot.kwh_true.sum() / tot.kwh_fc.sum() - 1),
        "coverage_country": float(np.mean((j.kwh_true >= j.lo) & (j.kwh_true <= j.hi))),
        "coverage_total": float(np.mean((t.kwh_true >= t.lo) & (t.kwh_true <= t.hi))),
        "nominal": res["interval"],
    }


def plan_value(cand: pd.DataFrame, truth: pd.DataFrame, chosen: list[tuple[int, int]], wcfg: dict[str, Any]) -> float:
    """What a plan really earns: true demand, true cannibalisation among the new sites, true capacity."""
    if not chosen:
        return 0.0
    ro = wcfg["rollout"]
    costs = economics.Costs.from_config(
        analysis()["rollout"], capacity_per_point=wcfg["network"]["types"]["HPC"]["cap_per_point"]
    )
    idx = np.array([i for i, _ in chosen])
    pts = np.array([k for _, k in chosen], dtype=float)
    lat, lon = cand.lat.to_numpy()[idx], cand.lon.to_numpy()[idx]
    d = sm.haversine(lat[:, None], lon[:, None], lat[None, :], lon[None, :])
    w = np.exp(-d / ro["cannibalisation_km"]) * (d < 3 * ro["cannibalisation_km"])
    np.fill_diagonal(w, 0.0)
    logm = truth.log_demand.to_numpy()[idx] - ro["cannibalisation"] * w.sum(axis=1)
    growth = np.array([wcfg["countries"][c]["growth"] for c in cand.country.to_numpy()[idx]])
    v = economics.npv(
        logm, pts, cand.planned_price_eur_kwh.to_numpy()[idx], cand.grid_kw.to_numpy()[idx], growth, costs
    )
    return float(v.sum())


def rollout_eval(res: dict[str, Any], truth_dir: Path, wcfg: dict[str, Any]) -> dict[str, Any]:
    cand = rollout.candidates_frame()
    truth = (
        pd.read_parquet(truth_dir / "candidates.parquet")
        .set_index("candidate_id")
        .reindex(cand.candidate_id)
        .reset_index()
    )
    pos = {c: i for i, c in enumerate(cand.candidate_id)}
    values = {}
    for name, plan in res["plans"].items():
        chosen = [(pos[s["candidate_id"]], s["points"]) for s in plan["selection"]]
        values[name] = {
            "true_npv_eur": plan_value(cand, truth, chosen, wcfg),
            "expected_npv_eur": plan["expected_npv_eur"],
        }
    # the oracle: the same programme, knowing every candidate's true demand and the true cannibalisation
    costs = economics.Costs.from_config(
        analysis()["rollout"], capacity_per_point=wcfg["network"]["types"]["HPC"]["cap_per_point"]
    )
    options = np.array(analysis()["rollout"]["points_options"], dtype=float)
    growth = np.array([wcfg["countries"][c]["growth"] for c in cand.country])
    V = economics.npv(
        truth.log_demand.to_numpy()[None, :, None],
        options[None, None, :],
        cand.planned_price_eur_kwh.to_numpy()[None, :, None],
        cand.grid_kw.to_numpy()[None, :, None],
        growth[None, :, None],
        costs,
    )
    keep = V[0].max(axis=1) > 0
    k8 = list(options).index(8)
    margin = V[0][:, k8] + economics.capex(np.full(len(cand), 8.0), cand.grid_kw.to_numpy(), costs)
    pairs = rollout._pairs(cand, keep, -wcfg["rollout"]["cannibalisation"], margin)
    oracle = rollout.optimise(cand, V, pairs, risk=False, keep=keep)
    best = plan_value(cand, truth, oracle["chosen"], wcfg)
    for v in values.values():
        v["regret_eur"] = best - v["true_npv_eur"]
        v["share_of_oracle"] = v["true_npv_eur"] / best if best else None
    # how well the site model ranks and sizes demand
    from scipy.stats import spearmanr

    pred = pd.DataFrame(res["candidate_npv"]).set_index("candidate_id").reindex(cand.candidate_id)
    rho = float(spearmanr(pred.npv_mean.to_numpy(), V[0][:, k8]).statistic)
    covered = float(np.mean((V[0][:, k8] >= pred.npv_p10.to_numpy()) & (V[0][:, k8] <= pred.npv_p90.to_numpy())))
    return {"oracle_npv_eur": best, "plans": values, "rank_correlation_npv": rho, "coverage_p10_p90": covered}


def run(results: dict[str, Any], truth_dir: Path = TRUTH, world: Path | None = None) -> dict[str, Any]:
    wcfg = (
        json.loads((truth_dir / "world_config.json").read_text())
        if (truth_dir / "world_config.json").exists()
        else load_config(world or ROOT / "config" / "world.yaml")
    )
    return {
        "price_change": price_change(results["price_change"], truth_dir),
        "trial": trial(results["trial"], truth_dir),
        "competitor": competitor(results["competitor"], truth_dir),
        "forecast": forecast(results["forecast"], truth_dir),
        "rollout": rollout_eval(results["rollout"], truth_dir, wcfg),
    }
