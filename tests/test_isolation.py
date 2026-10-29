"""Only the simulator and the evaluation may know the truth."""

import ast
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src" / "cndl"
ALLOWED = {"sim", "evaluation.py", "pipeline.py", "cli.py", "settings.py"}
FORBIDDEN_TEXT = ["world.yaml", "TRUTH", "build/truth", "world_config"]


def modules():
    for p in SRC.rglob("*.py"):
        rel = p.relative_to(SRC)
        if rel.parts[0] in ALLOWED:
            continue
        yield rel, p


def test_no_analysis_module_imports_the_simulator_or_the_evaluation():
    for rel, p in modules():
        tree = ast.parse(p.read_text())
        for node in ast.walk(tree):
            names = []
            if isinstance(node, ast.ImportFrom):
                names = [node.module or ""] + [a.name for a in node.names]
            elif isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            for n in names:
                assert "sim" not in n.split(".") and "evaluation" not in n.split("."), f"{rel} imports {n}"


def test_no_analysis_module_names_the_truth():
    for rel, p in modules():
        text = p.read_text()
        for word in FORBIDDEN_TEXT:
            assert word not in text, f"{rel} mentions {word}"
