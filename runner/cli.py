#!/usr/bin/env python3
"""Run workflow task(s) and write simulation-records/*.json"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from runner.record import run_task
from runner.tasks import TASKS


def main() -> int:
    parser = argparse.ArgumentParser(description="Run CUDA/CANN workflow tasks")
    parser.add_argument("ids", nargs="*", help="Task ids, e.g. cuda-s1-q1. Default: all")
    parser.add_argument("--list", action="store_true", help="List task ids")
    args = parser.parse_args()

    if args.list:
        for tid in sorted(TASKS):
            print(tid)
        return 0

    ids = args.ids or sorted(TASKS.keys())
    results = []
    for tid in ids:
        if tid not in TASKS:
            print(f"skip unknown: {tid}", file=sys.stderr)
            continue
        print(f"running {tid}...", flush=True)
        record = run_task(tid, TASKS[tid]["run"])
        results.append({"id": tid, "outcome": record["outcome"]})
        print(f"  -> {record['outcome'][:80]}")

    index = ROOT / "simulation-records" / "index.json"
    index.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nWrote {len(results)} record(s) to simulation-records/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
