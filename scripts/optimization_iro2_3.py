#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from ase.io import read


def analyze_results(outputs_dir: str = "outputs"):
    outdir = Path(outputs_dir)
    results_path = outdir / "results" / "optimization_results.json"
    if not results_path.exists():
        raise FileNotFoundError(f"Results JSON not found: {results_path}")

    data = json.loads(results_path.read_text())
    print("[analysis] Optimization summary")
    for k in [
        "method",
        "structure_in",
        "structure_out_xyz",
        "energy_eV_final",
        "fmax_eV_per_A_final",
        "elapsed_s",
        "atoms",
        "constrained_atoms_est",
    ]:
        if k in data:
            print(f"  - {k}: {data[k]}")
    return data


def compare_structures(a_path: str, b_path: str):
    a = read(a_path)
    b = read(b_path)
    if len(a) != len(b):
        print(f"[compare] Different atom counts: {len(a)} vs {len(b)}")
        return

    dr = a.positions - b.positions
    rmsd = float(np.sqrt((dr * dr).sum() / len(a)))
    dz_max = float(np.max(np.abs(dr[:, 2])))

    print("[compare] Structure comparison")
    print(f"  - A: {a_path}")
    print(f"  - B: {b_path}")
    print(f"  - RMSD (Å): {rmsd:.6f}")
    print(f"  - max |Δz| (Å): {dz_max:.6f}")


def traj_stats(traj_file: str):
    # Reads last frame only if multi-frame file; ASE read() reads last by default for traj.
    atoms = read(traj_file)
    print("[traj] Basic info")
    print(f"  - file: {traj_file}")
    print(f"  - atoms: {len(atoms)}")
    print(f"  - z-range (Å): {atoms.positions[:,2].min():.3f} .. {atoms.positions[:,2].max():.3f}")


def main():
    p = argparse.ArgumentParser(description="Analyze IrO2 optimization outputs.")
    p.add_argument("--outputs", default="outputs")
    p.add_argument("--compare", nargs=2, metavar=("A", "B"))
    p.add_argument("--traj")
    args = p.parse_args()

    if not args.compare and not args.traj:
        analyze_results(args.outputs)

    if args.compare:
        compare_structures(args.compare[0], args.compare[1])

    if args.traj:
        traj_stats(args.traj)


if __name__ == "__main__":
    main()
