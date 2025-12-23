import json
from pathlib import Path
from ase.io import read
import numpy as np

def analyze_result(result_dir: str = "results"):
    #Analyze optimization results and print summary
    result_path = Path(result_dir)
    
    # Find all result JSON files
    result_files = list(result_path.glob("*_results.json"))
    
    if not result_files:
        print(f"No results found in {result_dir}/")
        return
    
    print(f"\n{'='*70}")
    print(f"OPTIMIZATION RESULTS SUMMARY")
    print(f"{'='*70}\n")
    
    for result_file in sorted(result_files):
        with open(result_file) as f:
            data = json.load(f)
        
        print(f"File: {result_file.name}")
        print(f"  Structure:     {Path(data['structure_file']).name}")
        print(f"  Method:        {data['method']}")
        print(f"  Steps:         {data['n_steps']}/{data['max_steps']}")
        print(f"  Time:          {data['time_minutes']:.1f} min")
        print(f"  Energy:        {data['e_initial']:.4f} → {data['e_final']:.4f} eV")
        print(f"  ΔE:            {data['delta_e']:.4f} eV")
        print(f"  fmax:          {data['fmax_initial']:.4f} → {data['fmax_final']:.4f} eV/Å")
        print(f"  Converged:     {'YES' if data['converged'] else 'NO'}")
        print()


def compare_structures(initial_xyz: str, final_xyz: str):
    """
    Compare initial and final structures
    """
    initial = read(initial_xyz)
    final = read(final_xyz)
    
    # Find H atom (last atom)
    h_idx = len(initial) - 1
    
    h_init = initial[h_idx].position
    h_final = final[h_idx].position
    h_displacement = np.linalg.norm(h_final - h_init)
    
    print(f"\n{'='*70}")
    print(f"STRUCTURE COMPARISON")
    print(f"{'='*70}")
    print(f"Initial: {initial_xyz}")
    print(f"Final:   {final_xyz}")
    print(f"\nH atom displacement:")
    print(f"  Initial position: ({h_init[0]:.3f}, {h_init[1]:.3f}, {h_init[2]:.3f})")
    print(f"  Final position:   ({h_final[0]:.3f}, {h_final[1]:.3f}, {h_final[2]:.3f})")
    print(f"  Total movement:   {h_displacement:.3f} Å")
    print(f"{'='*70}\n")
    
    # Calculate RMSD for all atoms
    rmsd = np.sqrt(((initial.positions - final.positions)**2).sum(axis=1).mean())
    print(f"Overall RMSD: {rmsd:.4f} Å")


def visualize_trajectory(traj_file: str):
    """
    Quick stats from trajectory
    """
    from ase.io import read as read_traj
    
    traj = read_traj(traj_file, index=":")
    
    energies = [atoms.get_potential_energy() for atoms in traj]
    
    print(f"\n{'='*70}")
    print(f"TRAJECTORY ANALYSIS")
    print(f"{'='*70}")
    print(f"File: {traj_file}")
    print(f"Steps: {len(traj)}")
    print(f"Energy range: {min(energies):.4f} to {max(energies):.4f} eV")
    print(f"Total ΔE: {energies[-1] - energies[0]:.4f} eV")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze optimization results")
    parser.add_argument("--summary", action="store_true", help="Show all results summary")
    parser.add_argument("--compare", nargs=2, metavar=("INITIAL", "FINAL"), 
                        help="Compare two structures")
    parser.add_argument("--traj", help="Analyze trajectory file")
    parser.add_argument("--dir", default="results", help="Results directory")
    
    args = parser.parse_args()
    
    if args.summary or (not args.compare and not args.traj):
        analyze_result(args.dir)
    
    if args.compare:
        compare_structures(args.compare[0], args.compare[1])
    
    if args.traj:
        visualize_trajectory(args.traj)
