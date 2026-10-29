"""cndl run: the whole pipeline. cndl world / warehouse / analyse for the parts."""

from __future__ import annotations

from pathlib import Path

import typer

from . import pipeline, warehouse

app = typer.Typer(add_completion=False, help="Decisions for a charge-point network, checked against known truth.")


@app.command()
def run(world: Path = typer.Option(None, help="a world file: overrides of config/world.yaml")) -> None:
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
def heldout(name: str) -> None:
    """Run a held-out set of worlds once per world; the first run is kept as it is."""
    from .heldout import run_set

    out = run_set(name)
    typer.echo(out["summary"])


@app.command("briefs")
def briefs_cmd() -> None:
    """Rebuild the briefs from the last run's results."""
    import json

    from . import briefs
    from .settings import REPORTS, RESULTS

    results = {
        n: json.loads((RESULTS / f"{n}.json").read_text())
        for n in ["price_change", "trial", "competitor", "forecast", "rollout"]
    }
    b = briefs.build(results)
    pipeline.write_json(RESULTS / "briefs.json", b)
    briefs.write_markdown(b, REPORTS / "briefs")


@app.command("export-web")
def export_web() -> None:
    """Copy the results the decision room shows into web/public/data."""
    import shutil

    from .settings import RESULTS, ROOT

    out = ROOT / "web" / "public" / "data"
    out.mkdir(parents=True, exist_ok=True)
    for n in ["briefs", "price_change", "trial", "competitor", "forecast", "rollout", "evaluation"]:
        shutil.copy(RESULTS / f"{n}.json", out / f"{n}.json")
    typer.echo(f"copied to {out}")


@app.command()
def analyse() -> None:
    pipeline.analyse()


if __name__ == "__main__":
    app()
