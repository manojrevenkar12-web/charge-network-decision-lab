import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from cndl.analysis.economics import Costs, capex, npv, to_z
from cndl.settings import analysis

C = Costs.from_config(analysis()["rollout"], capacity_per_point=36.0)


def test_battery_only_when_the_grid_is_too_small():
    small = capex(np.array([8.0]), np.array([500.0]), C)[0]
    big = capex(np.array([8.0]), np.array([5000.0]), C)[0]
    assert small - big == pytest.approx(8 * C.battery_per_point)


@given(st.floats(0.5, 30), st.floats(0.5, 30))
def test_more_demand_is_worth_more(a, b):
    za, zb = to_z(np.array([a]), 36.0), to_z(np.array([b]), 36.0)
    va = npv(za, np.array([8.0]), np.array([0.8]), np.array([3000.0]), np.array([0.1]), C)[0]
    vb = npv(zb, np.array([8.0]), np.array([0.8]), np.array([3000.0]), np.array([0.1]), C)[0]
    assert (va - vb) * (a - b) >= -1e-6


@given(st.floats(0.1, 30))
def test_capacity_transform_round_trips(s):
    z = to_z(np.array([s]), 36.0)[0]
    back = 36.0 * (1 - np.exp(-np.exp(z) / 36.0))
    assert back == pytest.approx(min(s, 0.97 * 36.0), rel=1e-9)


def test_an_empty_site_loses_its_capex():
    v = npv(np.array([-20.0]), np.array([4.0]), np.array([0.8]), np.array([3000.0]), np.array([0.0]), C)[0]
    assert v < -capex(np.array([4.0]), np.array([3000.0]), C)[0]
