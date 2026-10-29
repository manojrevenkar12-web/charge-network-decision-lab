"""The release gate: thresholds from config/release.yaml applied to the evaluation."""

from __future__ import annotations

from typing import Any

import yaml

from .settings import ROOT


def check(ev: dict[str, Any], rules: dict[str, Any] | None = None) -> dict[str, Any]:
    r = rules or yaml.safe_load((ROOT / "config" / "release.yaml").read_text())
    checks: list[tuple[str, bool, str]] = []
    pc = ev["price_change"]
    checks.append(("price change: range covers the truth", pc["range_covers_truth"], f"truth {pc['truth_log']:.3f}"))
    if r["price_change"].get("usual_estimate_must_miss"):
        twfe = next(m for m in pc["methods"] if m["method"] == "twfe")
        checks.append(
            (
                "price change: two-way fixed effects misses (the trap is armed)",
                not twfe["covers"],
                f"twfe error {twfe['error']:+.3f}",
            )
        )
    tr = ev["trial"]
    covered = sum(1 for k in ["revenue_eur", "offpeak_share", "log_peak_sessions"] if tr[k]["covers"])
    checks.append(("trial: intervals cover the truth", covered >= r["trial"]["min_covered"], f"{covered} of 3"))
    if r["trial"]["decision_correct"]:
        checks.append(
            (
                "trial: decision matches the truth",
                tr["decision_correct"] is not False,
                f"true net {tr['true_net_value_per_site_day']:.0f} EUR",
            )
        )
    co = ev["competitor"]
    if co.get("available") and r["competitor"]["pooled_covers"]:
        checks.append(
            (
                "competitor: network-wide interval covers the truth",
                co["pooled"]["covers"],
                f"truth {co['pooled']['truth']:.3f}",
            )
        )
    fc = ev["forecast"]
    checks.append(
        (
            "forecast: country MAPE",
            fc["mape_country"] <= r["forecast"]["max_mape_country"],
            f"{fc['mape_country'] * 100:.1f}%",
        )
    )
    checks.append(
        (
            "forecast: interval coverage",
            fc["coverage_country"] >= r["forecast"]["min_coverage_country"],
            f"{fc['coverage_country'] * 100:.0f}%",
        )
    )
    ro = ev["rollout"]
    best = ro["plans"]["optimised_risk"]
    checks.append(
        (
            "rollout: share of the oracle's NPV",
            best["share_of_oracle"] >= r["rollout"]["min_share_of_oracle"],
            f"{best['share_of_oracle'] * 100:.0f}%",
        )
    )
    for other in r["rollout"]["must_beat"]:
        checks.append((f"rollout: beats {other}", best["true_npv_eur"] > ro["plans"][other]["true_npv_eur"], ""))
    return {
        "passed": all(ok for _, ok, _ in checks),
        "checks": [{"check": c, "passed": ok, "detail": d} for c, ok, d in checks],
    }
