from pathlib import Path
from ase.io import read, write
from ase import Atom
from ase.data import covalent_radii
from ase.constraints import FixAtoms
from ase.neighborlist import neighbor_list
import numpy as np
import json

def setup_structure(
    input_file: str = "inputs/slab_clean_2x2.in",
    o_index: int = 20,
    oh_distance: float = 1.0,
    z_freeze: float = 20.0,
    neighbor_cutoff: float = 1.5,
):
    # Create output directory
    Path("structures").mkdir(exist_ok=True)
    
    # Adjust H covalent radius
    covalent_radii[1] = 0.6
    
    print(f"Reading {input_file}...")
    slab = read(slab_clean_2x2, format="espresso-in")
    
    # Disable PBC for xTB
    slab.set_pbc((False, False, False))
    
    print(f"Initial structure: {len(slab)} atoms")
    print(f"Cell: {slab.cell.cellpar()}")
    
    # Find surface oxygen atoms
    positions = slab.get_positions()
    symbols = slab.get_chemical_symbols()
    z_top = positions[:, 2].max()
    
    surface_O = [
        i for i, s in enumerate(symbols)
        if s == "O" and (z_top - positions[i, 2]) < 1.5
    ]
    
    print(f"\nSurface O atoms (within 1.5 Å of top): {surface_O}")
    
    if o_index not in surface_O:
        print(f"  Warning: O[{o_index}] is not in surface list!")
        print(f"   O[{o_index}] is at z={positions[o_index, 2]:.2f} Å")
        print(f"   Top of slab: z={z_top:.2f} Å")
    
    # Add hydrogen atom
    O_pos = slab.positions[o_index]
    H_pos = O_pos + np.array([0.0, 0.0, oh_distance])
    slab.append(Atom("H", position=H_pos))
    h_index = len(slab) - 1
    
    print(f"\nAdded H atom:")
    print(f"  Index: {h_index}")
    print(f"  Position: {H_pos}")
    print(f"  Total atoms: {len(slab)}")
    
    # Freeze bottom layers
    z = slab.positions[:, 2]
    freeze_mask = z < z_freeze
    slab.set_constraint(FixAtoms(mask=freeze_mask))
    n_frozen = int(freeze_mask.sum())
    n_mobile = len(slab) - n_frozen
    
    print(f"\nFreezing atoms below z={z_freeze:.1f} Å:")
    print(f"  Frozen: {n_frozen} atoms")
    print(f"  Mobile: {n_mobile} atoms")
    
    # Check neighbors around H
    i, j, d = neighbor_list("ijd", slab, cutoff=[neighbor_cutoff] * len(slab))
    neighbors = [(int(jj), float(dd)) for ii, jj, dd in zip(i, j, d) if ii == h_index]
    
    print(f"\nNeighbors within {neighbor_cutoff} Å of H:")
    for jj, dd in neighbors:
        sym = slab[jj].symbol
        pos = slab[jj].position
        print(f"  {sym}[{jj}]: {dd:.3f} Å at ({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f})")
    
    # Save prepared structure
    output_file = f"structures/slab_H_o{o_index}_ready.xyz"
    write(output_file, slab)
    
    # Save metadata
    metadata = {
        "input_file": input_file,
        "o_index": o_index,
        "h_index": h_index,
        "oh_distance": oh_distance,
        "z_freeze": z_freeze,
        "n_atoms": len(slab),
        "n_frozen": n_frozen,
        "n_mobile": n_mobile,
        "neighbors": neighbors,
        "h_position": H_pos.tolist(),
        "o_position": O_pos.tolist(),
    }
    
    meta_file = f"structures/metadata_o{o_index}.json"
    with open(meta_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f" Structure prepared successfully!")
    print(f"{'='*60}")
    print(f"Output files:")
    print(f"  Structure: {output_file}")
    print(f"  Metadata:  {meta_file}")
    print(f"\nNext step:")
    print(f"  python run_optimization.py {output_file}")
    print(f"{'='*60}")
    
    return slab, metadata


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Prepare H* adsorption structure")
    parser.add_argument("--input", default="inputs/slab_clean_2x2.in", help="Input structure file")
    parser.add_argument("--o-index", type=int, default=20, help="Oxygen atom index for H placement")
    parser.add_argument("--oh-dist", type=float, default=1.0, help="O-H distance (Å)")
    parser.add_argument("--z-freeze", type=float, default=20.0, help="Freeze atoms below this z (Å)")
    
    args = parser.parse_args()
    
    slab, meta = setup_structure(
        input_file=args.input,
        o_index=args.o_index,
        oh_distance=args.oh_dist,
        z_freeze=args.z_freeze,
    )
