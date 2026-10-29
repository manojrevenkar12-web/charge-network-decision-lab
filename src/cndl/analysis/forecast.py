"""Question 4: how much energy will the existing network sell in the next twelve months, by country?

Three forecasters, chosen between on a backtest (the model fitted to data up to September 2025
forecasts October 2025 to September 2026, which has already happened):

- seasonal naive with year-on-year growth, the benchmark any model has to beat,
- exponential smoothing per country (statsforecast AutoETS),
- a global gradient-boosted model over all sites (mlforecast with LightGBM) that knows each
  site's age, so sites still ramping up after opening are not extrapolated like mature ones,
  summed to countries.

Intervals are conformal: the backtest's errors by horizon, scaled to the forecast, so an 80%
interval is as wide as the model's own recent mistakes say it should be.
"""

from __future__ import annotations

from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
from mlforecast import MLForecast
from statsforecast import StatsForecast
from statsforecast.models import AutoETS, SeasonalNaive

from ..settings import analysis
from ..warehouse import connect


def load() -> tuple[pd.DataFrame, pd.DataFrame]:
    con = connect()
    site = con.execute(
        """
        select m.site_id, m.country, m.month, m.kwh, m.kwh / nullif(m.sessions, 0) as kwh_per_session,
               s.segment, s.charger_type, s.points, s.opened
        from site_month m join sites s using (site_id)
        where m.days = extract(day from last_day(m.month)) -- complete months only
        """
    ).df()
    con.close()
    site["month"] = pd.to_datetime(site.month)
    site["opened"] = pd.to_datetime(site.opened)
    country = site.groupby(["country", "month"], as_index=False).kwh.sum()
    return site, country


def _months(start: pd.Timestamp, n: int) -> pd.DatetimeIndex:
    return pd.date_range(start, periods=n, freq="MS")


def seasonal_growth(country: pd.DataFrame, h: int) -> pd.DataFrame:
    out = []
    for c, g in country.groupby("country"):
        s = g.set_index("month").kwh.sort_index()
        last12, prev12 = s.iloc[-12:], s.iloc[-24:-12]
        growth = last12.sum() / prev12.sum()
        fut = _months(s.index[-1] + pd.offsets.MonthBegin(), h)
        vals = [last12.iloc[i % 12] * growth ** (1 + i // 12) for i in range(h)]
        out.append(pd.DataFrame({"country": c, "month": fut, "kwh": vals}))
    return pd.concat(out)


def ets(country: pd.DataFrame, h: int) -> pd.DataFrame:
    df = country.rename(columns={"country": "unique_id", "month": "ds", "kwh": "y"})
    df["y"] = np.log(df.y)
    sf = StatsForecast(models=[AutoETS(season_length=12, model="ZZA"), SeasonalNaive(season_length=12)], freq="MS")
    fc = sf.forecast(df=df, h=h)
    return pd.DataFrame({"country": fc.unique_id, "month": fc.ds, "kwh": np.exp(fc["AutoETS"])})


def global_gbm(site: pd.DataFrame, h: int) -> pd.DataFrame:
    """One model for every site: lags of its own log energy, its age and what kind of site it is."""
    df = site.rename(columns={"site_id": "unique_id", "month": "ds"}).copy()
    df["y"] = np.log1p(df.kwh)
    static = ["country", "segment", "charger_type", "points", "opened"]
    for col in ["country", "segment", "charger_type"]:
        df[col] = df[col].astype("category")
    df["opened_m"] = df.opened.dt.year * 12 + df.opened.dt.month
    df = df.drop(columns=["kwh", "opened"])

    def age(d: pd.DataFrame) -> pd.DataFrame:
        d["age"] = d.ds.dt.year * 12 + d.ds.dt.month - d.opened_m
        return d

    df = age(df)
    mf = MLForecast(
        models={
            "gbm": lgb.LGBMRegressor(
                n_estimators=400, learning_rate=0.05, num_leaves=31, min_child_samples=40, verbose=-1
            )
        },
        freq="MS",
        lags=[1, 2, 3, 12],
        date_features=["month"],
    )
    mf.fit(
        df[["unique_id", "ds", "y", "country", "segment", "charger_type", "points", "opened_m", "age"]],
        static_features=[*static[:-1], "opened_m"],
    )
    last = df.ds.max()
    fut = pd.DataFrame(
        [(u, m) for u in df.unique_id.unique() for m in _months(last + pd.offsets.MonthBegin(), h)],
        columns=["unique_id", "ds"],
    )
    fut = fut.merge(df.drop_duplicates("unique_id")[["unique_id", "opened_m"]], on="unique_id")
    fut = age(fut)
    fc = mf.predict(h=h, X_df=fut[["unique_id", "ds", "age"]])
    fc = fc.merge(site.drop_duplicates("site_id")[["site_id", "country"]], left_on="unique_id", right_on="site_id")
    fc["kwh"] = np.expm1(fc.gbm)
    return fc.groupby(["country", "ds"], as_index=False).kwh.sum().rename(columns={"ds": "month"})


# sessions a point can serve per day, from session length (the same domain knowledge as question 1)
CAPACITY_PER_POINT = {"HPC": 36.0, "DC": 24.0, "AC": 9.0}


def capacity_aware(site: pd.DataFrame, h: int, fit_months: int = 24) -> pd.DataFrame:
    """Each site's own growth path on a scale where a charger's capacity is linearised.

    Demand per point grows roughly exponentially until queues form; then it bends towards what
    the chargers can serve. On z = log(-cap * log(1 - s/cap)) that bend is undone, so a linear
    trend per site with a monthly pattern per segment can be extrapolated and transformed back.
    Sites with less than nine complete months use their last three months and the median trend
    of their segment and country."""
    d = site.copy()
    d["days"] = d.month.dt.days_in_month
    cap = d.charger_type.map(CAPACITY_PER_POINT)
    per_point_day = (d.kwh / d.kwh_per_session) / (d.days * d.points)
    s = np.minimum(per_point_day, 0.97 * cap)
    d["z"] = np.log(-cap * np.log(1 - s / cap))
    last = d.month.max()
    d = d[d.month > last - pd.DateOffset(months=fit_months)]
    d["t"] = (d.month.dt.year - last.year) * 12 + d.month.dt.month - last.month
    d["moy"] = d.month.dt.month
    # monthly pattern by segment from within-site deviations of a linear trend (two passes)
    season = pd.Series(0.0, index=pd.MultiIndex.from_product([d.segment.unique(), range(1, 13)]))
    for _ in range(3):
        d["zs"] = d.z - season.reindex(list(zip(d.segment, d.moy, strict=True))).to_numpy()
        coef = d.groupby("site_id")[["t", "zs"]].apply(
            lambda g: pd.Series(
                np.polyfit(g.t, g.zs, 1) if len(g) >= 9 else [np.nan, g.zs.iloc[-3:].mean()], index=["b", "a"]
            )
        )
        d = d.drop(columns=[c for c in ["b", "a"] if c in d]).join(coef, on="site_id")
        resid = d.z - (d.a + d.b.fillna(0) * d.t)
        season = resid.groupby([d.segment, d.moy]).mean()
        season = season - season.groupby(level=0).transform("mean")
    info = d.groupby("site_id").agg(
        a=("a", "first"),
        b=("b", "first"),
        segment=("segment", "first"),
        country=("country", "first"),
        charger_type=("charger_type", "first"),
        points=("points", "first"),
        kps=("kwh_per_session", "last"),
    )
    pooled = info.groupby(["country", "segment"]).b.median()
    info["b"] = info.b.fillna(
        pd.Series(pooled.reindex(list(zip(info.country, info.segment, strict=True))).to_numpy(), index=info.index)
    ).fillna(info.b.median())
    rows = []
    for k, m in enumerate(_months(last + pd.offsets.MonthBegin(), h), start=1):
        z = info.a + info.b * k + season.reindex(list(zip(info.segment, [m.month] * len(info), strict=True))).to_numpy()
        c = info.charger_type.map(CAPACITY_PER_POINT)
        spd = c * (1 - np.exp(-np.exp(z) / c))
        kwh = spd * info.points * m.days_in_month * info.kps
        rows.append(pd.DataFrame({"country": info.country, "month": m, "kwh": kwh}))
    return pd.concat(rows).groupby(["country", "month"], as_index=False).kwh.sum()


METHODS = {"seasonal_growth": "country", "ets": "country", "global_gbm": "site", "capacity_aware": "site"}


def _forecast(method: str, site: pd.DataFrame, country: pd.DataFrame, h: int) -> pd.DataFrame:
    if method == "seasonal_growth":
        return seasonal_growth(country, h)
    if method == "ets":
        return ets(country, h)
    if method == "capacity_aware":
        return capacity_aware(site, h)
    return global_gbm(site, h)


def run() -> dict[str, Any]:
    cfg = analysis()["forecast"]
    h = cfg["horizon_months"]
    site, country = load()
    last = country.month.max()
    cut = last - pd.DateOffset(months=cfg["backtest_months"])
    bt_site, bt_country = site[site.month <= cut], country[country.month <= cut]
    actual = country[country.month > cut].set_index(["country", "month"]).kwh
    backtest = {}
    errors = {}
    for m in METHODS:
        fc = _forecast(m, bt_site, bt_country, cfg["backtest_months"]).set_index(["country", "month"]).kwh
        j = pd.concat([fc.rename("fc"), actual.rename("actual")], axis=1).dropna()
        rel = np.log(j.actual / j.fc)
        tot = j.groupby(level=1).sum()
        backtest[m] = {
            "mape_country": float(np.mean(np.abs(j.actual / j.fc - 1))),
            "mape_total": float(np.mean(np.abs(tot.actual / tot.fc - 1))),
            "bias_total": float(tot.actual.sum() / tot.fc.sum() - 1),
        }
        errors[m] = rel
    best = min(backtest, key=lambda k: backtest[k]["mape_country"])
    fc = _forecast(best, site, country, h)
    # conformal interval: quantiles of the backtest's log errors, by horizon month
    rel = errors[best].reset_index()
    rel["horizon"] = rel.groupby("country").cumcount()
    a = (1 - cfg["interval"]) / 2
    q = rel.groupby("horizon")[0].quantile([a, 1 - a]).unstack()
    fc = fc.sort_values(["country", "month"]).copy()
    fc["horizon"] = fc.groupby("country").cumcount()
    fc = fc.join(q, on="horizon")
    fc["lo"] = fc.kwh * np.exp(fc[a])
    fc["hi"] = fc.kwh * np.exp(fc[1 - a])
    total = fc.groupby("month")[["kwh"]].sum()
    # the total's interval from the sum of the country errors at each horizon (they move together)
    tot_rel = rel.groupby(["horizon", "month"])[0].mean().groupby(level=0).quantile([a, 1 - a]).unstack()
    total["lo"] = total.kwh.to_numpy() * np.exp(tot_rel[a].reindex(range(len(total))).to_numpy())
    total["hi"] = total.kwh.to_numpy() * np.exp(tot_rel[1 - a].reindex(range(len(total))).to_numpy())
    history = country.groupby("month").kwh.sum()
    return {
        "question": "How much energy will the existing network sell in the next twelve months?",
        "backtest": backtest,
        "chosen": best,
        "interval": cfg["interval"],
        "forecast": [
            {
                "country": r.country,
                "month": str(r.month.date()),
                "kwh": float(r.kwh),
                "lo": float(r.lo),
                "hi": float(r.hi),
            }
            for r in fc.itertuples()
        ],
        "total": [
            {"month": str(m.date()), "kwh": float(r.kwh), "lo": float(r.lo), "hi": float(r.hi)}
            for m, r in total.iterrows()
        ],
        "history_total": [{"month": str(m.date()), "kwh": float(v)} for m, v in history.items()],
        "history_country_last12": country[country.month > last - pd.DateOffset(months=12)]
        .groupby("country", as_index=False)
        .kwh.sum()
        .to_dict("records"),
        "next_12_months_gwh": float(total.kwh.sum() / 1e6),
        "last_12_months_gwh": float(history.iloc[-12:].sum() / 1e6),
    }
