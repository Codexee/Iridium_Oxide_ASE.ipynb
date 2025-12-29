#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import os

import numpy as np
from ase.io import read
from ase.build import minimize_rotation_and_translation  # Added for alignment

def analyze_results(outputs_dir):
    results_path = f"results/slab_H_o20_ready_results.json"
    outdir = Path(results_path)
    if not outdir.exists():
        raise FileNotFoundError(f"Results JSON not found: {results_path}")

    with open(results_path) as f:
        data = json.load(f)

    result = {}

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
            result[k] = data[k]

    Path(outputs_dir).mkdir(parents=True, exist_ok=True)
    with open(f"{outputs_dir}/optimization.json", "w") as f:
        json.dump(result, f, indent=2)

    return data


def compare_structures(a_path: str, b_path: str):
    try:
        a = read(a_path)
        b = read(b_path)
    except Exception as e:
        raise ValueError(f"Error reading structures: {e}")

    if len(a) != len(b):
        print(f"[compare] Different atom counts: {len(a)} vs {len(b)}")
        return

    # Check if symbols match
    if a.get_chemical_symbols() != b.get_chemical_symbols():
        print("[compare] Warning: Atom symbols do not match—RMSD may be unreliable!")

    # Align structures
    minimize_rotation_and_translation(a, b)

    dr = a.positions - b.positions
    rmsd = float(np.sqrt((dr * dr).sum() / len(a)))
    dz_max = float(np.max(np.abs(dr[:, 2])))

    print("[compare] Structure comparison")
    print(f"  - A: {a_path}")
    print(f"  - B: {b_path}")
    print(f"  - RMSD (Å, after alignment): {rmsd:.6f}")
    print(f"  - max |Δz| (Å): {dz_max:.6f}")


def traj_stats(traj_file: str):
    try:
        # Try to read all frames for multi-frame trajectories
        traj = read(traj_file, index=':')
        if isinstance(traj, list):  # Multi-frame
            print(f"[traj] Multi-frame trajectory with {len(traj)} frames")
            z_mins = [atoms.positions[:, 2].min() for atoms in traj]
            z_maxs = [atoms.positions[:, 2].max() for atoms in traj]
            print(f"  - z-range across frames (Å): {min(z_mins):.3f} .. {max(z_maxs):.3f}")
            atoms = traj[-1]  # Use last for other stats
        else:  # Single frame
            atoms = traj
            print("[traj] Single-frame file")
    except Exception as e:
        # Fallback
        try:
            atoms = read(traj_file)
            print("[traj] Assuming single frame (or read error)")
        except Exception as e2:
            raise ValueError(f"Error reading trajectory: {e2}")

    print(f"  - file: {traj_file}")
    print(f"  - atoms: {len(atoms)}")
    print(f"  - z-range (last frame, Å): {atoms.positions[:,2].min():.3f} .. {atoms.positions[:,2].max():.3f}")


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
