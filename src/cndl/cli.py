"""cndl run: the whole pipeline. cndl world / warehouse / analyse for the parts."""

from __future__ import annotations

from pathlib import Path

import typer

from . import pipeline, warehouse

app = typer.Typer(add_completion=False, help="Decisions for a charge-point network, checked against known truth.")


@app.command()
def run(world: Path = typer.Option(None, help="a world file to use instead of config/world.yaml")) -> None:
    """World, warehouse, the five questions, briefs, evaluation and the release gate."""
    out = pipeline.run(world)
    for c in out["gate"]["checks"]:
        typer.echo(f"  {'ok  ' if c['passed'] else 'FAIL'} {c['check']} {c['detail']}")
    raise typer.Exit(0 if out["gate"]["passed"] else 1)


@app.command("world")
def world_cmd(world: Path = typer.Option(None)) -> None:
    typer.echo(pipeline.build_world(world))


@app.command("warehouse")
def warehouse_cmd() -> None:
    typer.echo(warehouse.build())


@app.command()
def analyse() -> None:
    pipeline.analyse()


if __name__ == "__main__":
    app()
