from pathlib import Path
from ase.io import read, write
from xtb.ase.calculator import XTB
from ase.optimize import BFGS
import numpy as np
import json
import time
from ase.constraints import FixAtoms


def _unique_path(path: Path) -> Path:
    """
    If path exists, append _1, _2, ... before suffix until it doesn't.
    Example: slab_opt.log -> slab_opt_1.log
    """
    if not path.exists():
        return path
    stem, suffix = path.stem, path.suffix
    parent = path.parent
    n = 1
    while True:
        candidate = parent / f"{stem}_{n}{suffix}"
        if not candidate.exists():
            return candidate
        n += 1


def run_optimization(
    structure_file: str,
    fmax: float = 0.05,
    max_steps: int = 250,
    method: str = "GFN2-xTB",
    output_dir: str = "results",
):
    outdir_path = Path(output_dir)
    outdir_path.mkdir(parents=True, exist_ok=True)

    start_time = time.time()

    print(f"{'='*70}")
    print(f"XTB GEOMETRY OPTIMIZATION")
    print(f"{'='*70}")
    print(f"Structure: {structure_file}")
    print(f"Method: {method}")
    print(f"Convergence: fmax < {fmax} eV/Å")
    print(f"Max steps: {max_steps}")
    print(f"Output dir: {outdir_path.resolve()}")
    print(f"{'='*70}\n")

    print("Loading structure...")
    slab = read(structure_file)

    # Check constraints
    n_constrained = 0
    if not slab.constraints:
        print("No constraints found. Re-applying FixAtoms by z-threshold...")
        z = slab.positions[:, 2]
        zmin = z.min()
        z_freeze = zmin + 5.0
        mask = z < z_freeze
        slab.set_constraint(FixAtoms(mask=mask))
        n_constrained = int(mask.sum())
        print(f" Applied FixAtoms: {n_constrained} frozen atoms (z < {z_freeze:.2f} Å)")

    print(f"Total atoms: {len(slab)}")
    print(f"Mobile atoms: {len(slab) - n_constrained}")

    print(f"\nSetting up {method} calculator...")
    slab.calc = XTB(method=method)

    print("\nCalculating initial state...")
    e_initial = slab.get_potential_energy()
    forces = slab.get_forces()
    fmax_initial = np.sqrt((forces**2).sum(axis=1)).max()

    print(f"  Initial energy: {e_initial:.6f} eV")
    print(f"  Initial fmax:   {fmax_initial:.6f} eV/Å")

    base_name = Path(structure_file).stem

    # Pre-compute output filenames (uniqued if they already exist)
    traj_file = _unique_path(outdir_path / f"{base_name}_opt.traj")
    log_file = _unique_path(outdir_path / f"{base_name}_opt.log")
    final_traj = _unique_path(outdir_path / f"{base_name}_final.traj")
    final_xyz = _unique_path(outdir_path / f"{base_name}_relaxed.xyz")
    results_file = _unique_path(outdir_path / f"{base_name}_results.json")

    if fmax_initial < fmax:
        print(f"\nAlready converged! No optimization needed.")
        write(final_traj, slab)
        write(final_xyz, slab)

        results = {
            "structure_file": structure_file,
            "method": method,
            "fmax_target": fmax,
            "max_steps": max_steps,
            "n_steps": 0,
            "converged": True,
            "initial": True,
            "e_initial": float(e_initial),
            "e_final": float(e_initial),
            "delta_e": 0.0,
            "fmax_initial": float(fmax_initial),
            "fmax_final": float(fmax_initial),
            "time_seconds": time.time() - start_time,
            "time_minutes": (time.time() - start_time) / 60,
            "base_name": base_name,
            "input_structure": str(Path(structure_file)),
            "trajectory_file": str(traj_file),
            "log_file": str(log_file),
            "final_traj": str(final_traj),
            "final_xyz": str(final_xyz),
            "results_file": str(results_file),
        }

        with open(results_file, "w") as f:
            json.dump(results, f, indent=2)

        print(f"Final structure saved to: {final_traj}")
        print(f"Results summary saved to: {results_file}")
        print(f"{'='*70}\n")
        return slab, results

    print(f"\n{'='*70}")
    print("Starting optimization...")
    print(f"{'='*70}")
    print(f"Trajectory: {traj_file}")
    print(f"Log file:   {log_file}")
    print(f"{'='*70}\n")

    schedule = [
        {"fmax": 0.20, "steps": 150, "maxstep": 0.10},
        {"fmax": 0.08, "steps": 200, "maxstep": 0.06},
        {"fmax": fmax, "steps": max_steps, "maxstep": 0.04},
    ]

    total_steps = 0
    stage_steps = 0

    for i, st in enumerate(schedule, 1):
        print(f"\n--- Stage {i}/{len(schedule)}: fmax={st['fmax']} steps={st['steps']} maxstep={st['maxstep']} ---")
        opt = BFGS(slab, trajectory=str(traj_file), logfile=str(log_file), maxstep=st["maxstep"])
        opt.run(fmax=st["fmax"], steps=st["steps"])
        stage_steps = opt.get_number_of_steps()
        total_steps += stage_steps

        forces_now = slab.get_forces()
        fmax_now = float(np.sqrt((forces_now**2).sum(axis=1)).max())
        print(f"Stage {i} done: fmax_now={fmax_now:.6f} eV/Å")

        if fmax_now <= fmax:
            print("Reached final target early, stopping schedule.")
            break

    e_final = slab.get_potential_energy()
    forces_final = slab.get_forces()
    fmax_final = np.sqrt((forces_final**2).sum(axis=1)).max()
    converged = bool(fmax_final <= fmax)
    elapsed_time = time.time() - start_time

    print(f"\n{'='*70}")
    print("OPTIMIZATION COMPLETE")
    print(f"{'='*70}")
    print(f"Steps taken:        {total_steps}")
    print(f"Time elapsed:       {elapsed_time:.1f} seconds ({elapsed_time/60:.1f} min)")
    print(f"Initial energy:     {e_initial:.6f} eV")
    print(f"Final energy:       {e_final:.6f} eV")
    print(f"Energy change:      {e_final - e_initial:.6f} eV")
    print(f"Initial fmax:       {fmax_initial:.6f} eV/Å")
    print(f"Final fmax:         {fmax_final:.6f} eV/Å")
    print(f"Converged:          {'YES' if converged else 'NO'}")
    print(f"{'='*70}")

    write(final_traj, slab)
    write(final_xyz, slab)
    print(f"\nFinal structure saved to: {final_traj}")

    results = {
        "structure_file": structure_file,
        "method": method,
        "fmax_target": fmax,
        "max_steps": max_steps,
        "n_steps": total_steps,
        "converged": converged,
        "e_initial": float(e_initial),
        "e_final": float(e_final),
        "delta_e": float(e_final - e_initial),
        "fmax_initial": float(fmax_initial),
        "fmax_final": float(fmax_final),
        "time_seconds": elapsed_time,
        "time_minutes": elapsed_time / 60,
        "base_name": base_name,
        "input_structure": str(Path(structure_file)),
        "trajectory_file": str(traj_file),
        "log_file": str(log_file),
        "final_traj": str(final_traj),
        "final_xyz": str(final_xyz),
        "results_file": str(results_file),
        "n_steps_total": total_steps,
        "n_steps_last_stage": stage_steps,
    }

    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Results summary saved to: {results_file}")
    print(f"{'='*70}\n")

    return slab, results


def quick_test(structure_file: str, steps: int = 5):
    print(f"\n{'='*70}")
    print(f"QUICK TEST MODE - {steps} STEPS ONLY")
    print(f"{'='*70}\n")
    return run_optimization(
        structure_file,
        fmax=0.05,
        max_steps=steps,
        output_dir="test_results",
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run xTB optimization")
    parser.add_argument("structure", help="Prepared structure file (.traj)")
    parser.add_argument("--fmax", type=float, default=0.05, help="Force convergence (eV/Å)")
    parser.add_argument("--steps", type=int, default=250, help="Maximum steps")
    parser.add_argument(
        "--method",
        default="GFN2-xTB",
        choices=["GFN1-xTB", "GFN2-xTB"],
        help="xTB method (GFN1 is faster)",
    )
    parser.add_argument("--test", action="store_true", help="Quick test mode (5 steps)")
    parser.add_argument("--output", default="results", help="Output directory")

    args = parser.parse_args()

    if args.test:
        slab, results = quick_test(args.structure, steps=5)
    else:
        slab, results = run_optimization(
            args.structure,
            fmax=args.fmax,
            max_steps=args.steps,
            method=args.method,
            output_dir=args.output,
        )

    print("\nDone!")
