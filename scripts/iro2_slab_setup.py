#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from ase import Atom
from ase.constraints import FixAtoms
from ase.io import read, write
from ase.neighborlist import neighbor_list


def setup_structure(
    input_file: str = "inputs/slab_clean_2x2.in",
    o_index: int = 20,
    oh_distance: float = 1.0,
    z_freeze: float = 20.0,
    neighbor_cutoff: float = 1.5,
    outputs_dir: str = "outputs",
):
    """
    Read slab, place H above a chosen O, freeze atoms below z_freeze,
    and write outputs for the next step.
    """
    input_path = Path(input_file)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    slab = read(str(input_path))

    n0 = len(slab)
    if not (0 <= o_index < n0):
        raise IndexError(f"o_index={o_index} out of range for {n0} atoms (0..{n0-1})")

    # Slab z-extent (for diagnostics)
    zmax = slab.positions[:, 2].max()

    # Position of selected O
    o_pos = slab.positions[o_index].copy()

    # Place H above O along +z
    h_pos = o_pos + np.array([0.0, 0.0, float(oh_distance)])

    # Sanity check: ensure H is above the slab top
    if h_pos[2] <= zmax:
        print(
            f"[warn] H z={h_pos[2]:.3f} Å is not above slab top zmax={zmax:.3f} Å. "
            "Raising H to be above slab."
        )
    h_pos[2] = zmax + float(oh_distance)

    slab.append(Atom("H", position=h_pos))
    h_index = len(slab) - 1

    slab.append(Atom("H", position=h_pos))
    h_index = len(slab) - 1

    # Freeze atoms below z_freeze
    freeze_mask = slab.positions[:, 2] < float(z_freeze)
    slab.set_constraint(FixAtoms(mask=freeze_mask))

    # Neighbors of H (for sanity check)
    i, j, d = neighbor_list("ijd", slab, cutoff=float(neighbor_cutoff))
    neigh = [(int(jj), float(dd)) for ii, jj, dd in zip(i, j, d) if int(ii) == h_index]
    neigh_sorted = sorted(neigh, key=lambda x: x[1])

    outdir = Path(outputs_dir)
    (outdir / "results").mkdir(parents=True, exist_ok=True)

    # Write structures
    write(str(outdir / "slab_with_H.traj"), slab)
    write(str(outdir / "slab_with_H.xyz"), slab)

    meta = {
        "input_file": str(input_path),
        "num_atoms_initial": int(n0),
        "num_atoms_total": int(len(slab)),
        "o_index": int(o_index),
        "h_index": int(h_index),
        "oh_distance_A": float(oh_distance),
        "z_freeze_A": float(z_freeze),
        "neighbor_cutoff_A": float(neighbor_cutoff),
        "o_position_A": [float(x) for x in o_pos],
        "h_position_A": [float(x) for x in h_pos],
        "num_frozen": int(np.count_nonzero(freeze_mask)),
        "neighbors_of_H": [{"index": idx, "distance_A": dist} for idx, dist in neigh_sorted],
    }
    (outdir / "results" / "slab_setup_metadata.json").write_text(json.dumps(meta, indent=2))

    print(f"[slab_setup] Loaded {input_path} (atoms={n0})")
    print(f"[slab_setup] Placed H at index {h_index} above O index {o_index} by {oh_distance} Å")
    print(f"[slab_setup] Frozen atoms: {meta['num_frozen']} (z < {z_freeze} Å)")
    print(f"[slab_setup] H neighbors within {neighbor_cutoff} Å: {neigh_sorted}")

    return slab, meta


def main():
    p = argparse.ArgumentParser(description="IrO2 slab setup: place H above O, freeze bottom layers, write outputs/")
    p.add_argument("--input", default="inputs/slab_clean_2x2.in")
    p.add_argument("--o-index", type=int, default=20, help="0-based index of O atom to adsorb H onto")
    p.add_argument("--oh-dist", type=float, default=1.0)
    p.add_argument("--z-freeze", type=float, default=20.0)
    p.add_argument("--neighbor-cutoff", type=float, default=1.5)
    p.add_argument("--outputs", default="outputs")
    args = p.parse_args()

    setup_structure(
        input_file=args.input,
        o_index=args.o_index,
        oh_distance=args.oh_dist,
        z_freeze=args.z_freeze,
        neighbor_cutoff=args.neighbor_cutoff,
        outputs_dir=args.outputs,
    )


if __name__ == "__main__":
    main()
