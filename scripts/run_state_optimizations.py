#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Tuple


def run_one(optimizer: str, traj: Path, outdir: Path) -> Tuple[str, int, str]:
    """
    Run optimizer on one trajectory.
    Returns (traj_path, return_code, last_output_snippet).
    """
    cmd = [sys.executable, optimizer, str(traj), "--output", str(outdir)]
    p = subprocess.run(cmd, capture_output=True, text=True)
    # Keep the tail for debugging
    tail = ""
    if p.stdout:
        tail += p.stdout[-2000:]
    if p.stderr:
        tail += ("\n" + p.stderr[-2000:])
    return (str(traj), p.returncode, tail)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outputs", default="outputs", help="Root outputs dir")
    ap.add_argument(
        "--optimizer",
        default="scripts/optimization_iro2_test.py",
        help="Optimizer script path",
    )
    ap.add_argument(
        "--jobs",
        type=int,
        default=max(1, (os.cpu_count() or 2) // 2),
        help="Number of parallel optimizations to run (default: half cores).",
    )
    ap.add_argument("--dry_run", action="store_true", help="Print commands but do not run")
    args = ap.parse_args()

    out_root = Path(args.outputs)
    trajs: List[Path] = sorted(out_root.glob("states/**/structures/*.traj"))
    if not trajs:
        raise FileNotFoundError(f"No .traj found under {out_root}/states/**/structures")

    # Prepare jobs
    jobs = []
    for traj in trajs:
        res_dir = traj.parent.parent / "results"
        res_dir.mkdir(parents=True, exist_ok=True)
        jobs.append((traj, res_dir))

    print(f"Found {len(jobs)} structures. Running with --jobs {args.jobs}")

    if args.dry_run:
        for traj, res_dir in jobs:
            cmd = [sys.executable, args.optimizer, str(traj), "--output", str(res_dir)]
            print(" ".join(cmd))
        return

    failures = 0
    # ThreadPool is fine: work happens in external subprocesses.
    with ThreadPoolExecutor(max_workers=max(1, args.jobs)) as ex:
        futs = [ex.submit(run_one, args.optimizer, traj, res_dir) for traj, res_dir in jobs]
        for fut in as_completed(futs):
            traj_path, rc, tail = fut.result()
            if rc != 0:
                failures += 1
                print(f"\n[FAIL] {traj_path}\n{tail}\n")
            else:
                print(f"[ok] {traj_path}")

    if failures:
        raise SystemExit(f"{failures} optimization(s) failed.")
    print(f"\nDone. Optimized {len(jobs)} structures.")


if __name__ == "__main__":
    main()
