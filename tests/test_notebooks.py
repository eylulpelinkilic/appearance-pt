"""Notebooks are generated from scripts; they must be current and must not embed secrets."""

import ast
import json

from conftest import ROOT, load_script

mk = load_script("scripts/make_notebooks.py")


def test_notebooks_match_generator():
    for name, nb in mk.build().items():
        path = ROOT / name
        assert path.exists(), f"{name} missing: run python scripts/make_notebooks.py"
        assert path.read_text() == mk.render(nb), f"{name} is stale: run python scripts/make_notebooks.py"


def test_notebook_code_cells_are_valid_python():
    for name in mk.build():
        nb = json.loads((ROOT / name).read_text())
        for cell in nb["cells"]:
            if cell["cell_type"] == "code":
                ast.parse("".join(cell["source"]), filename=f"{name}:{cell['id']}")
