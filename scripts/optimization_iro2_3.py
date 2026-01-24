#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re

import numpy as np
from ase.io import read
from ase.build import minimize_rotation_and_translation


def _safe_stem(s: str) -> str:
    # Keep filenames safe and short-ish
    s = s.strip()
    s = re.sub(r"[^A-Za-z0-9_.-]+", "_", s)
    return s[:180] if len(s) > 180 else s


def analyze_all_results(outputs_dir: str):
    outdir = Path(outputs_dir)
    results_dir = outdir / "results"

    candidates = sorted(results_dir.glob("*_results.json"))
    if not candidates:
        raise FileNotFoundError(f"No *_results.json found under {results_dir}")

    keymap = {
        "method": "method",
        "structure_in": "structure_file",
        "structure_out_xyz": "final_xyz",
        "structure_out_traj": "final_traj",
        "energy_eV_final": "e_final",
        "energy_eV_initial": "e_initial",
        "fmax_eV_per_A_final": "fmax_final",
        "fmax_eV_per_A_initial": "fmax_initial",
        "elapsed_s": "time_seconds",
        "steps": "n_steps",
        "converged": "converged",
    }

    index = []

    print(f"[analysis] Found {len(candidates)} result file(s) in {results_dir}")

    for results_path in candidates:
        data = json.loads(results_path.read_text())

        # Prefer deriving stem from the input structure filename (more stable)
        structure_in = str(data.get("structure_file", "unknown_structure"))
        stem = _safe_stem(Path(structure_in).stem)

        # Fallback: use the results file stem
        if stem in ("", "unknown_structure"):
            stem = _safe_stem(results_path.stem.replace("_results", ""))

        result = {}
        for outk, ink in keymap.items():
            if ink in data:
                result[outk] = data[ink]

        # Write per-structure summaries (no overwrites across different structures)
        opt_path = results_dir / f"optimization_{stem}.json"
        opt_full_path = results_dir / f"optimization_full_{stem}.json"

        opt_path.write_text(json.dumps(result, indent=2))
        opt_full_path.write_text(json.dumps(data, indent=2))

        index.append(
            {
                "stem": stem,
                "results_json": str(results_path),
                "optimization_json": str(opt_path),
                "optimization_full_json": str(opt_full_path),
                "energy_eV_final": result.get("energy_eV_final"),
                "fmax_eV_per_A_final": result.get("fmax_eV_per_A_final"),
                "converged": result.get("converged"),
            }
        )

        print(f"[analysis] Wrote: {opt_path.name}, {opt_full_path.name}")

    # Write an index for easy downstream consumption
    index_path = results_dir / "optimization_index.json"
    index_path.write_text(json.dumps(index, indent=2))
    print(f"[analysis] Wrote index: {index_path.name}")

    return index


def compare_structures(a_path: str, b_path: str):
    a = read(a_path)
    b = read(b_path)

    if len(a) != len(b):
        print(f"[compare] Different atom counts: {len(a)} vs {len(b)}")
        return

    if a.get_chemical_symbols() != b.get_chemical_symbols():
        print("[compare] Warning: Atom symbols do not match—RMSD may be unreliable!")

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
    traj = read(traj_file, index=":")
    if isinstance(traj, list):
        print(f"[traj] Multi-frame trajectory with {len(traj)} frames")
        z_mins = [atoms.positions[:, 2].min() for atoms in traj]
        z_maxs = [atoms.positions[:, 2].max() for atoms in traj]
        print(f"  - z-range across frames (Å): {min(z_mins):.3f} .. {max(z_maxs):.3f}")
        atoms = traj[-1]
    else:
        atoms = traj
        print("[traj] Single-frame file")

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
        analyze_all_results(args.outputs)

    if args.compare:
        compare_structures(args.compare[0], args.compare[1])

    if args.traj:
        traj_stats(args.traj)


if __name__ == "__main__":
    main()
