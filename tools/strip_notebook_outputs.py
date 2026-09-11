#!/usr/bin/env python3
"""Clear cell outputs and execution counts from a Jupyter notebook in place.

Why this exists: cell outputs can bake in real rows, totals, or figures from
the source data, and this repo is public. Run this on any notebook before
committing it. Usage:

    python3 tools/strip_notebook_outputs.py notebooks/01_data_cleaning.ipynb [more.ipynb ...]
"""
import json
import sys


def strip(path):
    with open(path, encoding="utf-8") as f:
        nb = json.load(f)
    for cell in nb.get("cells", []):
        if cell.get("cell_type") == "code":
            cell["outputs"] = []
            cell["execution_count"] = None
    with open(path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
        f.write("\n")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("usage: strip_notebook_outputs.py <notebook.ipynb> [...]")
    for p in sys.argv[1:]:
        strip(p)
        print("stripped:", p)
