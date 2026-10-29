import duckdb
import pytest

from cndl.warehouse import ContractError, check


def test_a_broken_promise_stops_the_pipeline(monkeypatch):
    con = duckdb.connect()
    con.execute(
        "create table site_day as select 'A' as site_id, date '2026-01-01' as date, -1 as sessions, "
        "1.0 as kwh, 1.0 as revenue_eur, 0.5 as price_eur_kwh, 1.0 as availability"
    )
    for t in ["sites", "site_week", "trial_day", "candidates"]:
        con.execute(f"create table {t} as select 1 as x where false")
    monkeypatch.setattr(
        "cndl.warehouse.contracts", lambda: {"site_day": {"checks": ["sessions >= 0"], "unique": ["site_id", "date"]}}
    )
    with pytest.raises(ContractError, match="sessions >= 0"):
        check(con)
