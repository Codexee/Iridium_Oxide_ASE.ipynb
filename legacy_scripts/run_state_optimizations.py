#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import List


def main() -> None:
    ap = argparse.ArgumentParser(description="Run xTB optimizations sequentially over generated state .traj files.")
    ap.add_argument("--outputs", default="outputs", help="Root outputs dir (contains states/**/structures/*.traj)")
    ap.add_argument("--optimizer", default="scripts/optimization_iro2_test.py", help="Optimizer script path")
    ap.add_argument("--dry_run", action="store_true", help="Print commands but do not run them")
    ap.add_argument(
        "--exclude",
        default="",
        help="Comma-separated substrings to skip (e.g. hydrated). Leave empty to run all.",
    )
    args = ap.parse_args()

    out_root = Path(args.outputs)
    trajs: List[Path] = sorted(out_root.glob("states/**/structures/*.traj"))
    if not trajs:
        raise FileNotFoundError(f"No .traj found under {out_root}/states/**/structures")

    excludes = [s.strip() for s in args.exclude.split(",") if s.strip()]
    if excludes:
        trajs = [t for t in trajs if not any(ex in str(t) for ex in excludes)]

    if not trajs:
        raise FileNotFoundError("No .traj left to run after applying --exclude filter.")

    print(f"Found {len(trajs)} structures to optimize (sequential).", flush=True)

    failures = 0
    for idx, traj in enumerate(trajs, start=1):
        res_dir = traj.parent.parent / "results"
        res_dir.mkdir(parents=True, exist_ok=True)

        cmd = [sys.executable, args.optimizer, str(traj), "--output", str(res_dir)]
        print(f"\n[{idx}/{len(trajs)}] START {traj}", flush=True)
        print(" ".join(cmd), flush=True)

        if args.dry_run:
            continue

        p = subprocess.run(cmd, text=True)
        if p.returncode != 0:
            failures += 1
            print(f"[{idx}/{len(trajs)}] FAIL {traj} (exit={p.returncode})", flush=True)
            # Continue so we still get partial results
        else:
            print(f"[{idx}/{len(trajs)}] OK   {traj}", flush=True)

    if failures:
        raise SystemExit(f"{failures} optimization(s) failed. Partial results may still exist.")
    print("\nAll optimizations completed successfully.", flush=True)


if __name__ == "__main__":
    main()
