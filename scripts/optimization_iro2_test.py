from pathlib import Path
from ase.io import read, write
from xtb.ase.calculator import XTB
from ase.optimize import BFGS
import numpy as np
import json
import time
from ase.constraints import FixAtoms

def run_optimization(
    structure_file: str,
    fmax: float = 0.05,
    max_steps: int = 250,
    method: str = "GFN2-xTB",
    output_dir: str = "results",
):
    # Create output directory
    #Path(output_dir).mkdir(parents=True, exist_ok=True)  # Added parents=True
    outdir_path = Path(output_dir)
    outdir_path.mkdir(parents=True, exist_ok=True)

    # Start timing
    start_time = time.time()

    print(f"{'='*70}")
    print(f"XTB GEOMETRY OPTIMIZATION")
    print(f"{'='*70}")
    print(f"Structure: {structure_file}")
    print(f"Method: {method}")
    print(f"Convergence: fmax < {fmax} eV/Å")
    print(f"Max steps: {max_steps}")
    print(f"{'='*70}\n")

    # Load structure
    print("Loading structure...")
    slab = read(structure_file)

    # Check constraints
    n_constrained = 0  # Initialize to avoid NameError
    if not slab.constraints:
        print("No constraints found. Re-applying FixAtoms by z-threshold...")
        z = slab.positions[:, 2]
        zmin = z.min()
        z_freeze = zmin + 5.0  # freeze bottom 2 Å as an example
        mask = z < z_freeze
        slab.set_constraint(FixAtoms(mask=mask))
        n_constrained = int(mask.sum())
        print(f" Applied FixAtoms: {n_constrained} frozen atoms (z < {z_freeze:.2f} Å)")

    print(f"Total atoms: {len(slab)}")
    print(f"Mobile atoms: {len(slab) - n_constrained}")

    # Setup xTB calculator
    print(f"\nSetting up {method} calculator...")
    slab.calc = XTB(method=method)

    # Initial energy and forces
    print("\nCalculating initial state...")
    e_initial = slab.get_potential_energy()
    forces = slab.get_forces()
    fmax_initial = np.sqrt((forces**2).sum(axis=1)).max()

    print(f"  Initial energy: {e_initial:.6f} eV")
    print(f"  Initial fmax:   {fmax_initial:.6f} eV/Å")

    if fmax_initial < fmax:
        print(f"\n Already converged! No optimization needed.")
        final_file = outdir_path / f"{Path(structure_file).stem}_final.traj"
        write(final_file, slab)
        return slab, {
            "converged": True,
            "initial": True,
            "e_initial": float(e_initial),
            "e_final": float(e_initial),
            "fmax_initial": float(fmax_initial),
            "fmax_final": float(fmax_initial),
            "n_steps": 0,
            "time_seconds": time.time() - start_time,
        }

    # Setup output files
    #outdir = Path(output_dir)
    #outdir.mkdir(parents=True, exist_ok=True)
    base_name = Path(structure_file).stem
    traj_file = outdir_path / f"{base_name}_opt.traj"
    log_file = outdir_path / f"{base_name}_opt.log"

    print(f"\n{'='*70}")
    print(f"Starting optimization...")
    print(f"{'='*70}")
    print(f"Trajectory: {traj_file}")
    print(f"Log file:   {log_file}")
    print(f"{'='*70}\n")

    # Run optimization
    opt = BFGS(slab, trajectory=str(traj_file), logfile=str(log_file), maxstep=0.04)
    print(f"DEBUG: calling opt.run(fmax={fmax}, steps={max_steps})")
    try:
        opt.run(fmax=fmax, steps=max_steps)
        # ASE version compatibility: converged can be a method or an attribute
        #conv = getattr(opt, "converged", False)
        #converged = bool(conv() if callable(conv) else conv)
        #converged = None
    except Exception as e:
        print(f"\n  Optimization failed: {e}")
        converged = False
    print(f"DEBUG: opt.get_number_of_steps() -> {opt.get_number_of_steps()}")

    # Get final results
    e_final = slab.get_potential_energy()
    forces_final = slab.get_forces()
    fmax_final = np.sqrt((forces_final**2).sum(axis=1)).max()
    converged = bool(fmax_final <= fmax)
    n_steps = opt.get_number_of_steps()
    elapsed_time = time.time() - start_time

    # Print results
    print(f"\n{'='*70}")
    print(f"OPTIMIZATION COMPLETE")
    print(f"{'='*70}")
    print(f"Steps taken:        {n_steps}")
    print(f"Time elapsed:       {elapsed_time:.1f} seconds ({elapsed_time/60:.1f} min)")
    print(f"Initial energy:     {e_initial:.6f} eV")
    print(f"Final energy:       {e_final:.6f} eV")
    print(f"Energy change:      {e_final - e_initial:.6f} eV")
    print(f"Initial fmax:       {fmax_initial:.6f} eV/Å")
    print(f"Final fmax:         {fmax_final:.6f} eV/Å")
    print(f"Converged:          {'YES' if converged else 'NO'}")
    print(f"{'='*70}")

    final_traj = outdir_path / f"{base_name}_final.traj"
    write(final_traj, slab)
    print(f"\nFinal structure saved to: {final_traj}")

    # Maybe keep xyz for visualization convenience
    final_xyz = outdir_path / f"{base_name}_relaxed.xyz"
    write(final_xyz, slab)

    # Save results summary
    results = {
        "structure_file": structure_file,
        "method": method,
        "fmax_target": fmax,
        "max_steps": max_steps,
        "n_steps": n_steps,
        "converged": converged,
        "e_initial": float(e_initial),
        "e_final": float(e_final),
        "delta_e": float(e_final - e_initial),
        "fmax_initial": float(fmax_initial),
        "fmax_final": float(fmax_final),
        "time_seconds": elapsed_time,
        "time_minutes": elapsed_time / 60,
    }

    results_file = outdir_path / f"{base_name}_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"Results summary saved to: {results_file}")
    print(f"{'='*70}\n")

    return slab, results


# Quick test with limited steps
def quick_test(structure_file: str, steps: int = 5):
    print(f"\n{'='*70}")
    print(f"QUICK TEST MODE - {steps} STEPS ONLY")
    print(f"{'='*70}\n")

    return run_optimization(
        structure_file,
        fmax=0.05,
        max_steps=steps,
        output_dir="test_results"
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run xTB optimization")
    parser.add_argument("structure", help="Prepared structure file (.traj)")
    parser.add_argument("--fmax", type=float, default=0.05, help="Force convergence (eV/Å)")
    parser.add_argument("--steps", type=int, default=250, help="Maximum steps")
    parser.add_argument("--method", default="GFN2-xTB", choices=["GFN1-xTB", "GFN2-xTB"],
                        help="xTB method (GFN1 is faster)")
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

    print("\n Done!")
