"""Runs a held-out set of worlds (heldout/<set>/worlds/*.yaml), each in its own build directory and
process, and records the evaluation. The first run of a set is written to
reports/heldout/<set>/first_run.json and never rewritten; later runs go to latest.json.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from .settings import ROOT


def run_set(name: str, log: Any = print) -> dict[str, Any]:
    worlds = sorted((ROOT / "heldout" / name / "worlds").glob("*.yaml"))
    if not worlds:
        raise SystemExit(f"no worlds in heldout/{name}/worlds")
    out: dict[str, Any] = {"set": name, "started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "worlds": {}}
    for w in worlds:
        build = ROOT / "build" / "heldout" / name / w.stem
        reports = build / "reports"
        env = {**os.environ, "CNDL_BUILD": str(build), "CNDL_REPORTS": str(reports)}
        t0 = time.perf_counter()
        p = subprocess.run(
            [sys.executable, "-m", "cndl.cli", "run", "--world", str(w)], env=env, capture_output=True, text=True
        )
        rec: dict[str, Any] = {"seconds": round(time.perf_counter() - t0), "exit": p.returncode}
        latest = reports / "latest.json"
        if latest.exists():
            rec["evaluation"] = json.loads(latest.read_text())["evaluation"]
            rec["gate"] = json.loads((reports / "gate.json").read_text())
        else:
            rec["error"] = (p.stderr or p.stdout)[-2000:]
        out["worlds"][w.stem] = rec
        log(
            f"{w.stem}: {'gate pass' if rec.get('gate', {}).get('passed') else 'gate fail' if 'gate' in rec else 'error'} ({rec['seconds']} s)"
        )
    out["summary"] = summarise(out["worlds"])
    dest = ROOT / "reports" / "heldout" / name
    dest.mkdir(parents=True, exist_ok=True)
    text = json.dumps(out, indent=2) + "\n"
    (dest / "latest.json").write_text(text)
    first = dest / "first_run.json"
    if not first.exists():
        first.write_text(text)
    return out


def summarise(worlds: dict[str, Any]) -> dict[str, Any]:
    ok = [w["evaluation"] for w in worlds.values() if "evaluation" in w]
    n = len(ok)

    def share(f: Any) -> float | None:
        vals = [bool(f(e)) for e in ok if f(e) is not None]
        return sum(vals) / len(vals) if vals else None

    usual = [m for e in ok for m in e["price_change"]["methods"] if m["method"] == "twfe"]
    return {
        "worlds": len(worlds),
        "completed": n,
        "gate_passed": sum(1 for w in worlds.values() if w.get("gate", {}).get("passed")),
        "price_change_range_covers": share(lambda e: e["price_change"]["range_covers_truth"]),
        "price_change_decision_correct": share(lambda e: e["price_change"]["decision_correct"]),
        "twfe_covers": sum(m["covers"] for m in usual) / len(usual) if usual else None,
        "trial_effects_covered": share(lambda e: e["trial"]["revenue_eur"]["covers"]),
        "trial_decision_correct": share(lambda e: e["trial"]["decision_correct"]),
        "competitor_pooled_covers": share(lambda e: e["competitor"].get("pooled", {}).get("covers")),
        "forecast_mape_country": [round(e["forecast"]["mape_country"], 4) for e in ok],
        "forecast_coverage_country": [round(e["forecast"]["coverage_country"], 3) for e in ok],
        "rollout_share_of_oracle": [round(e["rollout"]["plans"]["optimised_risk"]["share_of_oracle"], 3) for e in ok],
        "rollout_beats_traffic": share(
            lambda e: (
                e["rollout"]["plans"]["optimised_risk"]["true_npv_eur"]
                > e["rollout"]["plans"]["highest_traffic"]["true_npv_eur"]
            )
        ),
    }
