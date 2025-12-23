#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from ase.io import read, write
from ase.optimize import BFGS
from xtb.ase.calculator import XTB


def run_optimization(
    structure_file: str,
    fmax: float = 0.05,
    max_steps: int = 200,
    method: str = "GFN2-xTB",
    outputs_dir: str = "outputs",
):
    outdir = Path(outputs_dir)
    (outdir / "results").mkdir(parents=True, exist_ok=True)

    structure_path = Path(structure_file)
    if not structure_path.exists():
        raise FileNotFoundError(f"Structure file not found: {structure_path}")

    slab = read(str(structure_path))

    # Count constrained atoms (best-effort)
    n_constrained = 0
    if slab.constraints:
        for c in slab.constraints:
            try:
                n_constrained += len(c.get_indices())
            except Exception:
                pass

    print(f"[opt] Loaded: {structure_path} atoms={len(slab)} constrained~={n_constrained}")

    # xTB calculator
    slab.calc = XTB(method=method)

    # Run BFGS
    traj_path = outdir / "slab_with_H_opt.traj"
    log_path = outdir / "slab_with_H_opt.log"

    start = time.time()
    opt = BFGS(slab, trajectory=str(traj_path), logfile=str(log_path))
    opt.run(fmax=float(fmax), steps=int(max_steps))
    elapsed = time.time() - start

    # Compute final energy/forces
    energy = float(slab.get_potential_energy())
    forces = slab.get_forces()
    fmax_final = float(np.linalg.norm(forces, axis=1).max()) if len(forces) else float("nan")

    # Write final structure
    write(str(outdir / "slab_with_H_opt.xyz"), slab)

    results = {
        "structure_in": str(structure_path),
        "structure_out_traj": str(traj_path),
        "structure_out_xyz": str(outdir / "slab_with_H_opt.xyz"),
        "logfile": str(log_path),
        "method": method,
        "fmax_target": float(fmax),
        "max_steps": int(max_steps),
        "elapsed_s": float(elapsed),
        "energy_eV_final": energy,
        "fmax_eV_per_A_final": fmax_final,
        "atoms": int(len(slab)),
        "constrained_atoms_est": int(n_constrained),
    }

    (outdir / "results" / "optimization_results.json").write_text(json.dumps(results, indent=2))

    print(f"[opt] Done. E_final={energy:.6f} eV  fmax_final={fmax_final:.6f} eV/Å  time={elapsed:.1f}s")
    return slab, results


def main():
    p = argparse.ArgumentParser(description="Run xTB optimization on an ASE structure file.")
    p.add_argument("--structure", default="outputs/slab_with_H.traj", help="Input structure (traj/xyz/in/etc)")
    p.add_argument("--fmax", type=float, default=0.05)
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--method", default="GFN2-xTB")
    p.add_argument("--outputs", default="outputs")
    args = p.parse_args()

    run_optimization(
        structure_file=args.structure,
        fmax=args.fmax,
        max_steps=args.steps,
        method=args.method,
        outputs_dir=args.outputs,
    )


if __name__ == "__main__":
    main()
