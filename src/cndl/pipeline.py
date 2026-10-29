"""The whole run: world, warehouse, the five questions, the briefs, the evaluation and the gate."""

from __future__ import annotations

import json
import time
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from . import briefs, gate, warehouse
from .analysis import competitor, forecast, offpeak_trial, price_change, rollout
from .settings import DATA, REPORTS, RESULTS, TRUTH, WAREHOUSE


def _json(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _json(v) for k, v in obj.items()}
    if isinstance(obj, list | tuple):
        return [_json(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, pd.Timestamp):
        return str(obj.date())
    return obj


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json(obj), indent=2) + "\n")


def build_world(
    world_file: Path | None = None, overrides: dict[str, Any] | None = None, data: Path = DATA, truth: Path = TRUTH
) -> dict[str, int]:
    import yaml

    from .settings import ROOT
    from .sim.world import build, load_config, write

    # a world file holds only what differs from config/world.yaml
    extra = yaml.safe_load(Path(world_file).read_text()) if world_file else {}
    cfg = load_config(ROOT / "config" / "world.yaml", {**(extra or {}), **(overrides or {})})
    return write(build(cfg), data, truth)


def analyse(results: Path = RESULTS, log: Any = print) -> dict[str, Any]:
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)
    out: dict[str, Any] = {}
    steps = [
        ("price_change", price_change.run),
        ("trial", offpeak_trial.run),
        ("competitor", competitor.run),
        ("forecast", forecast.run),
    ]
    for name, fn in steps:
        t0 = time.perf_counter()
        out[name] = fn()
        log(f"  {name}: {time.perf_counter() - t0:.0f} s")
    # the rollout builds on questions 1 and 4: the price effect and each country's growth
    pc = out["price_change"]
    shift = float(np.mean(pc["range_point_log"]))
    fc = pd.DataFrame(out["forecast"]["forecast"]).groupby("country").kwh.sum()
    hist = pd.DataFrame(out["forecast"]["history_country_last12"]).set_index("country").kwh
    growth = (fc / hist).to_dict()
    t0 = time.perf_counter()
    out["rollout"] = rollout.run(shift, growth)
    log(f"  rollout: {time.perf_counter() - t0:.0f} s")
    for name, r in out.items():
        write_json(results / f"{name}.json", r)
    return out


def run(
    world_file: Path | None = None, results: Path = RESULTS, reports: Path = REPORTS, log: Any = print
) -> dict[str, Any]:
    t0 = time.perf_counter()
    log("world")
    counts = build_world(world_file)
    log("warehouse")
    tables = warehouse.build(DATA, WAREHOUSE)
    log("analyses")
    out = analyse(results, log)
    log("briefs")
    b = briefs.build(out)
    write_json(results / "briefs.json", b)
    briefs.write_markdown(b, reports / "briefs")
    log("evaluation")
    from . import evaluation

    ev = evaluation.run(out, TRUTH)
    summary = {
        "generated_by": "cndl run",
        "seconds": round(time.perf_counter() - t0),
        "rows": {**counts, **{f"warehouse.{k}": v for k, v in tables.items()}},
        "evaluation": ev,
    }
    write_json(reports / "latest.json", summary)
    write_json(results / "evaluation.json", ev)
    verdict = gate.check(ev)
    write_json(reports / "gate.json", verdict)
    log(f"gate: {'pass' if verdict['passed'] else 'FAIL'}")
    return {"results": out, "evaluation": ev, "gate": verdict}
