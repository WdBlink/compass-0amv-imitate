#!/usr/bin/env python3
"""Run one autoresearch experiment and log it.

    python3 experiment.py --spec '{"window":55,"decay":0.925}' --desc 'text'

Prints the full diagnostic JSON on stdout and a machine-readable

    METRIC <value>

line on stderr, then appends one row to results.tsv.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

import numpy as np

from evaluate import evaluate
from model import build_panel

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results.tsv")
HEADER = ["iteration", "commit", "metric", "delta", "status", "description"]


def git_head() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                       cwd=HERE, text=True).strip()
    except Exception:
        return "-"


def read_last_metric() -> float | None:
    if not os.path.exists(RESULTS):
        return None
    last = None
    with open(RESULTS, encoding="utf-8") as fh:
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 4 and parts[0].isdigit():
                last = float(parts[2])
    return last


def append_row(iteration: int, commit: str, metric: float, delta: float,
               status: str, description: str, extra: dict) -> None:
    new = not os.path.exists(RESULTS)
    with open(RESULTS, "a", encoding="utf-8") as fh:
        if new:
            fh.write("\t".join(HEADER) + "\n")
        fh.write("\t".join([
            str(iteration), commit, f"{metric:.6f}", f"{delta:+.6f}", status,
            description.replace("\t", " "),
        ]) + "\n")
        fh.write("#detail " + json.dumps(extra, ensure_ascii=False) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True, help="JSON model spec")
    ap.add_argument("--desc", default="")
    ap.add_argument("--iteration", type=int, default=None)
    ap.add_argument("--status", default="keep")
    ap.add_argument("--no-log", action="store_true")
    args = ap.parse_args()

    spec = json.loads(args.spec)
    panel = build_panel()
    res = evaluate(spec, panel)

    prev = read_last_metric()
    delta = (res["metric"] - prev) if prev is not None else float("nan")

    extra = {
        "spec": spec,
        "mae_dev": round(res["mae_dev"], 6),
        "mae_holdout": round(res["mae_holdout"], 6),
        "rmse": round(res["rmse"], 6),
        "bias": round(res["bias"], 6),
        "dir_hit": round(res["dir_hit"], 4),
        "level_mape_pct": round(res["level_mape_pct"], 4),
        "trigger_match": f"{res['trigger_match']}/{res['trigger_n']}",
    }
    print(json.dumps(extra, ensure_ascii=False))
    print(f"METRIC {res['metric']:.6f}", file=sys.stderr)

    if not args.no_log:
        it = args.iteration
        if it is None:
            it = 0
            if os.path.exists(RESULTS):
                rows = [l for l in open(RESULTS, encoding="utf-8") if l[:1].isdigit()]
                it = len(rows)
        append_row(it, git_head(), res["metric"], delta, args.status, args.desc, extra)


if __name__ == "__main__":
    main()
