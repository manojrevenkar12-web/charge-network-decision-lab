"""The simulator on a small world: deterministic, and the traps it is meant to set are set."""

import numpy as np
import pytest

from cndl.sim.world import build, load_config

SMALL = {"network": {"sites": 260}, "rollout": {"candidates": 120}, "offpeak_trial": {"sites": 12, "weeks": 2}}


@pytest.fixture(scope="module")
def world():
    return build(load_config(overrides=SMALL))


def test_same_seed_same_world(world):
    again = build(load_config(overrides=SMALL))
    assert world.observed["daily"].sessions.sum() == again.observed["daily"].sessions.sum()


def test_the_analyses_never_see_the_hidden_columns(world):
    assert not {"quality", "local_trend", "elasticity"} & set(world.observed["sites"].columns)


def test_prices_went_up_where_demand_was_growing(world):
    params = world.truth["site_params"].set_index("site_id")
    raised = set(world.truth["price_change_att"].site_id)
    trend = params.local_trend
    assert trend[trend.index.isin(raised)].mean() > trend[~trend.index.isin(raised)].mean()


def test_the_price_increase_costs_sessions(world):
    assert world.truth["price_change_summary"]["att_log"] < 0


def test_trial_assignment_is_a_fair_coin(world):
    share = world.observed["trial_assignment"].discount.mean()
    assert 0.35 < share < 0.65
    assert np.isfinite(world.truth["trial"]["revenue_effect"])
