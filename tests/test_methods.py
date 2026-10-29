"""The estimators on panels where the answer is known by construction."""

import numpy as np
import pandas as pd
import pytest

from cndl.analysis import impute, sdid


def panel(n=120, w=80, effect=-0.1, trend_sd=0.0, select_on_trend=0.0, seed=0):
    rng = np.random.default_rng(seed)
    trend = rng.normal(0, trend_sd, n) if trend_sd else np.zeros(n)
    score = select_on_trend * (trend - trend.mean()) / (trend.std() or 1)
    treated = rng.random(n) < 1 / (1 + np.exp(-score))
    first = np.where(treated, rng.choice([40, 50, 60], n), -1)
    t = np.arange(w)
    week = rng.normal(0, 0.05, w)
    y = rng.normal(3, 0.3, n)[:, None] + trend[:, None] * t / 52 + week[None, :] + rng.normal(0, 0.03, (n, w))
    on = (first[:, None] >= 0) & (t[None, :] >= first[:, None])
    y = y + effect * on
    rows = [
        (f"s{i}", pd.Timestamp("2024-01-01") + pd.Timedelta(weeks=int(k)), y[i, k], int(on[i, k]), "g")
        for i in range(n)
        for k in range(w)
    ]
    df = pd.DataFrame(rows, columns=["site_id", "week", "y", "treat", "segment"])
    return df, y, first


def test_parallel_trends_imputation_recovers_the_effect():
    df, _, _ = panel(effect=-0.1)
    p = impute.to_panel(df, "y", "segment")
    y0, usable = impute.fit_untreated(p, trend=False)
    m = p.treated & usable[:, None]
    assert np.mean((p.y - y0)[m]) == pytest.approx(-0.1, abs=0.01)


def test_trend_imputation_recovers_the_effect_when_treatment_follows_growth():
    df, _, _ = panel(effect=-0.1, trend_sd=0.15, select_on_trend=2.0)
    p = impute.to_panel(df, "y", "segment")
    y0, _ = impute.fit_untreated(p, trend=False)
    naive = np.mean((p.y - y0)[p.treated])
    y0t, usable = impute.fit_untreated(p, trend=True)
    good = np.mean((p.y - y0t)[p.treated & usable[:, None]])
    assert good == pytest.approx(-0.1, abs=0.015)
    assert naive > good + 0.02  # ignoring the trends hides part of the loss


def test_synthetic_did_recovers_the_effect_under_parallel_trends():
    _, y, first = panel(effect=-0.08)
    est, cohorts = sdid.staggered(y, first)
    assert est == pytest.approx(-0.08, abs=0.01)
    assert len(cohorts) == 3


def test_simplex_weights_are_a_distribution():
    rng = np.random.default_rng(1)
    A = rng.normal(size=(30, 10))
    w = sdid._fw_simplex(A, A[:, 3], zeta=1e-3)
    assert w.min() >= -1e-12
    assert w.sum() == pytest.approx(1.0)
    assert w.argmax() == 3
