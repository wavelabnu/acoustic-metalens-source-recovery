"""NC Source Data: one CSV per graphical panel, written next to the figure."""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np


def write(out_dir: Path, name: str, **cols) -> Path:
    """Write `out_dir/source_data/<name>.csv`; columns may differ in length (padded blank)."""
    sd = Path(out_dir) / "source_data"
    sd.mkdir(parents=True, exist_ok=True)
    arrs = [np.atleast_1d(np.asarray(v, dtype=object)) for v in cols.values()]
    n = max(len(a) for a in arrs)
    p = sd / f"{name}.csv"
    with open(p, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(list(cols))
        for i in range(n):
            w.writerow(["" if i >= len(a) else _fmt(a[i]) for a in arrs])
    return p


def _fmt(v):
    if isinstance(v, (float, np.floating)):
        return f"{float(v):.6g}"
    if isinstance(v, (np.integer,)):
        return int(v)
    return v
