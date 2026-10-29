"""Loads the raw files into DuckDB, builds the marts in SQL and checks the contracts."""

from __future__ import annotations

from pathlib import Path

import duckdb

from .settings import DATA, ROOT, WAREHOUSE, analysis, contracts


class ContractError(RuntimeError):
    pass


def build(data: Path = DATA, db: Path = WAREHOUSE) -> dict[str, int]:
    db.parent.mkdir(parents=True, exist_ok=True)
    db.unlink(missing_ok=True)
    con = duckdb.connect(str(db))
    hours = ",".join(str(h) for h in analysis()["offpeak_trial"]["hours"])
    for f in sorted((ROOT / "sql").glob("*.sql")):
        con.execute(f.read_text().replace("{data}", str(data)).replace("{offpeak_hours}", hours))
    check(con)
    counts = {t: con.execute(f"select count(*) from {t}").fetchone()[0] for t in contracts()}
    con.close()
    return counts


def check(con: duckdb.DuckDBPyConnection) -> None:
    problems = []
    for table, c in contracts().items():
        for col in c.get("not_null", []):
            n = con.execute(f"select count(*) from {table} where {col} is null").fetchone()[0]
            if n:
                problems.append(f"{table}.{col}: {n} nulls")
        for rule in c.get("checks", []):
            n = con.execute(f"select count(*) from {table} where not ({rule})").fetchone()[0]
            if n:
                problems.append(f"{table}: {n} rows break '{rule}'")
        if key := c.get("unique"):
            cols = ", ".join(key)
            n = con.execute(
                f"select count(*) from (select {cols} from {table} group by all having count(*) > 1)"
            ).fetchone()[0]
            if n:
                problems.append(f"{table}: {n} duplicate keys ({cols})")
    if problems:
        raise ContractError("; ".join(problems))


def connect(db: Path = WAREHOUSE, read_only: bool = True) -> duckdb.DuckDBPyConnection:
    return duckdb.connect(str(db), read_only=read_only)
