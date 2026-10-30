"""Recompute what scripts/baseline.py recorded and report the differences: the data folders'
goal poses, map poses and plan lengths within TOLERANCE, the evaluation exactly. Exits 1 on any.

    uv run scripts/check_baseline.py
"""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np
from baseline import OUT, record

TOLERANCE = 1e-6


def differences(old, new, tolerance: float, where: str = "") -> list[str]:
    """Where ``old`` and ``new`` (JSON values) differ: numbers by more than ``tolerance``."""
    if isinstance(old, dict) and isinstance(new, dict):
        out = [
            f"{where}/{k}: only in {'baseline' if k in old else 'now'}" for k in old.keys() ^ new
        ]
        for k in old.keys() & new.keys():
            out += differences(old[k], new[k], tolerance, f"{where}/{k}")
        return out
    if isinstance(old, list) and isinstance(new, list) and len(old) == len(new):
        return [
            d
            for i, (a, b) in enumerate(zip(old, new, strict=True))
            for d in differences(a, b, tolerance, f"{where}[{i}]")
        ]
    numbers = (int, float)
    if isinstance(old, numbers) and isinstance(new, numbers) and not isinstance(old, bool):
        if old == new or abs(old - new) <= tolerance:
            return []
    elif old == new:
        return []
    return [f"{where}: {old!r} -> {new!r}"]


def main() -> None:
    argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    ).parse_args()
    baseline = json.loads((OUT / "baseline.json").read_text())
    summary, poses = record("check")
    found = []
    for name, now in poses.items():
        with np.load(OUT / f"poses_{name}.npz") as data:
            then = [data[f"arr_{r}"] for r in range(len(data.files))]
        if len(then) != len(now) or any(a.shape != b.shape for a, b in zip(then, now, strict=True)):
            found.append(
                f"/data/{name}/poses: shapes {[a.shape for a in then]} -> {[b.shape for b in now]}"
            )
            continue
        worst = max(float(np.abs(a - b).max()) for a, b in zip(then, now, strict=True))
        if worst > TOLERANCE:
            found.append(f"/data/{name}/poses: max abs difference {worst:.3g}")
        summary["data"][name].pop("poses_sha256")
        baseline["data"][name].pop("poses_sha256")
    found += differences(baseline["data"], summary["data"], TOLERANCE, "/data")
    found += differences(baseline["eval"], summary["eval"], 0.0, "/eval")
    print("\n".join(found) if found else "no differences from the baseline")
    sys.exit(1 if found else 0)


if __name__ == "__main__":
    main()
