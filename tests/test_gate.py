import copy

from cndl.gate import check

EV = {
    "price_change": {
        "range_covers_truth": True,
        "truth_log": -0.08,
        "methods": [{"method": "twfe", "covers": False, "error": 0.1}],
    },
    "trial": {
        "revenue_eur": {"covers": True},
        "offpeak_share": {"covers": False},
        "log_peak_sessions": {"covers": True},
        "decision_correct": True,
        "true_net_value_per_site_day": -100.0,
    },
    "competitor": {"available": True, "pooled": {"covers": True, "truth": -0.15}},
    "forecast": {"mape_country": 0.04, "coverage_country": 0.7},
    "rollout": {
        "plans": {
            "optimised_risk": {"share_of_oracle": 0.85, "true_npv_eur": 10.0},
            "highest_traffic": {"true_npv_eur": 5.0},
        }
    },
}


def test_a_good_evaluation_passes():
    assert check(EV)["passed"]


def test_a_disarmed_trap_fails():
    ev = copy.deepcopy(EV)
    ev["price_change"]["methods"][0]["covers"] = True
    assert not check(ev)["passed"]


def test_a_plan_worse_than_traffic_ranking_fails():
    ev = copy.deepcopy(EV)
    ev["rollout"]["plans"]["highest_traffic"]["true_npv_eur"] = 20.0
    assert not check(ev)["passed"]
