#!/usr/bin/env python3
"""
Check the loss table in README.md against the notebooks.

    python tools/check_notebooks.py              # compare the README with the committed outputs
    python tools/check_notebooks.py --execute    # re-run every notebook first (hours on a CPU)

Each notebook prints one line per epoch, `epoch N | train loss X` (the extended notebook prints
`epoch N | train X | val Y`). The script reads those lines from each notebook's stored output,
takes the epoch count and the first and last losses, and compares them with the "Results across
all stages" table in README.md, to the precision the table quotes (a leading ~ means rounded).

With --execute the notebooks are first run top to bottom with nbconvert into build/notebooks/,
and those fresh outputs are checked instead, within a 5% tolerance. Only the extended notebook
seeds PyTorch, and even its seed repeats a run only on the same device and PyTorch version: its
stored outputs come from a GPU, which draws its dropout masks from a different generator than a
CPU does.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
NOTEBOOKS = {
    "01": "notebook_01.ipynb",
    "02": "notebook_02.ipynb",
    "03": "notebook_03.ipynb",
    "04": "notebook_04.ipynb",
    "05": "notebook_05.ipynb",
    "06": "notebook_06.ipynb",
    "06_h23yonsei": "notebook_06_h23yonsei.ipynb",
}


def losses(nb_path: Path) -> tuple[int, list[float], list[float]]:
    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    text = "".join("".join(o.get("text", [])) for c in nb["cells"] if c["cell_type"] == "code"
                   for o in c.get("outputs", []))
    rows = re.findall(r"epoch\s+(\d+) \| train(?: loss)? ([\d.]+)(?: \| val ([\d.]+))?", text)
    epochs = int(rows[-1][0]) + 1 if rows else 0      # some notebooks print every fifth epoch
    train = [float(r[1]) for r in rows]
    val = [float(r[2]) for r in rows if r[2]]
    return epochs, train, val


def readme_table() -> dict[str, dict[str, str]]:
    text = (REPO / "README.md").read_text(encoding="utf-8")
    section = re.split(r"(?m)^## \d+\. Results across all stages", text, maxsplit=1)[1]
    section = section.split("\n## ", 1)[0]
    rows = {}
    for line in section.splitlines():
        cells = [c.strip().strip("*").replace("—", "-") for c in line.strip().strip("|").split("|")]
        if len(cells) == 7 and cells[0] in NOTEBOOKS:
            rows[cells[0]] = dict(initial=cells[3], final=cells[4], val=cells[5], epochs=cells[6])
    return rows


def matches(quoted: str, value: float | None, tolerance: float) -> bool:
    if quoted in ("-", ""):
        return value is None
    if value is None:
        return False
    approx = quoted.startswith("~")
    q = quoted.lstrip("~")
    decimals = len(q.split(".")[1]) if "." in q else 0
    if tolerance:
        return abs(value - float(q)) <= tolerance * float(q)
    return round(value, decimals) == float(q) if approx else f"{value:.{decimals}f}" == q


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--execute", action="store_true", help="re-run the notebooks before checking")
    args = ap.parse_args()

    source = REPO
    if args.execute:
        source = REPO / "build" / "notebooks"
        source.mkdir(parents=True, exist_ok=True)
        for name in NOTEBOOKS.values():
            print(f"executing {name} ...", flush=True)
            subprocess.run([sys.executable, "-m", "nbconvert", "--to", "notebook", "--execute",
                            "--ExecutePreprocessor.timeout=-1", "--output-dir", str(source),
                            str(REPO / name)], cwd=REPO, check=True)

    table = readme_table()
    failed = 0
    for key, name in NOTEBOOKS.items():
        row = table.get(key)
        if row is None:
            print(f"FAIL  {key:13s} no row in the README table")
            failed += 1
            continue
        epochs, train, val = losses(source / name)
        if not train:
            print(f"FAIL  {key:13s} no epoch lines in {name}")
            failed += 1
            continue
        tol = 0.05 if args.execute else 0.0
        checks = [
            ("epochs", row["epochs"], str(epochs), row["epochs"] == str(epochs)),
            ("first", row["initial"], f"{train[0]:.4f}", matches(row["initial"], train[0], tol)),
            ("final", row["final"], f"{train[-1]:.4f}", matches(row["final"], train[-1], tol)),
            ("val", row["val"], f"{val[-1]:.4f}" if val else "-",
             matches(row["val"], val[-1] if val else None, tol)),
        ]
        ok = all(c[3] for c in checks)
        failed += not ok
        detail = ", ".join(f"{n} {got} (README {q})" for n, q, got, _ in checks)
        print(f"{'PASS' if ok else 'FAIL'}  {key:13s} {detail}")

    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
