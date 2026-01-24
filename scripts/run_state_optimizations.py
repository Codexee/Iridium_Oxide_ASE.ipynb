#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outputs", default="outputs", help="Root outputs dir")
    ap.add_argument("--optimizer", default="scripts/optimization_iro2_test.py", help="Optimizer script path")
    ap.add_argument("--dry_run", action="store_true", help="Print commands but do not run")
    args = ap.parse_args()

    out_root = Path(args.outputs)
    trajs = sorted(out_root.glob("states/**/structures/*.traj"))
    if not trajs:
        raise FileNotFoundError(f"No .traj found under {out_root}/states/**/structures")

    for traj in trajs:
        # results folder sibling of structures
        res_dir = traj.parent.parent / "results"
        res_dir.mkdir(parents=True, exist_ok=True)

        cmd = ["python", args.optimizer, str(traj), "--output", str(res_dir)]
        print(" ".join(cmd))
        if not args.dry_run:
            subprocess.run(cmd, check=True)

    print(f"\nDone. Optimised {len(trajs)} structures.")


if __name__ == "__main__":
    main()

