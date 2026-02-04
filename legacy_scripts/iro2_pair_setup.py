#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Tuple

import numpy as np
from ase import Atom
from ase.constraints import FixAtoms
from ase.io import read, write
from ase.neighborlist import neighbor_list


# Default candidate sites you provided
DEFAULT_O_SITES = [21, 22, 23, 44, 46, 47, 69, 71]

# The compact pair set we agreed (shell-covering, efficient)
DEFAULT_PAIRS: List[Tuple[int, int]] = [
    # Ultra-short (1.517 Å)
    (44, 46),
    (69, 71),
    (21, 23),
    (21, 47),

    # ~3.0–3.56 Å
    (23, 47),
    (22, 46),
    (46, 47),
    (22, 23),
    (21, 22),
    (21, 46),
    (46, 69),
    (22, 69),

    # ~4.43–4.55 Å
    (23, 46),
    (22, 47),
    (21, 44),
    (46, 71),
    (22, 44),
]


def _surface_oxygen_indices(slab, surface_threshold: float):
    positions = slab.get_positions()
    symbols = slab.get_chemical_symbols()
    z_top = positions[:, 2].max()
    surface_O = [
        i for i, s in enumerate(symbols)
        if s == "O" and (z_top - positions[i, 2]) < surface_threshold
    ]
    return surface_O, float(z_top)


def prepare_pair_structure(
    slab0,
    o_i: int,
    o_j: int,
    oh_distance: float,
    z_freeze: float,
    neighbor_cutoff: float,
    surface_threshold: float,
):
    slab = slab0.copy()

    surface_O, z_top = _surface_oxygen_indices(slab, surface_threshold=surface_threshold)
    warnings = []
    for o_idx in (o_i, o_j):
        if o_idx not in surface_O:
            warnings.append(f"O[{o_idx}] not in surface-O list by threshold {surface_threshold} Å")

    # Place two H atoms
    oi_pos = slab.positions[o_i]
    oj_pos = slab.positions[o_j]
    hi_pos = oi_pos + np.array([0.0, 0.0, oh_distance])
    hj_pos = oj_pos + np.array([0.0, 0.0, oh_distance])

    slab.append(Atom("H", position=hi_pos))
    h_i_index = len(slab) - 1
    slab.append(Atom("H", position=hj_pos))
    h_j_index = len(slab) - 1

    # Freeze bottom layers (same idea as your existing setup)
    z = slab.positions[:, 2]
    freeze_mask = z < z_freeze
    slab.set_constraint(FixAtoms(mask=freeze_mask))

    # Neighbor info (optional, but nice for debugging)
    i, j, d = neighbor_list("ijd", slab, cutoff=neighbor_cutoff)
    neigh_i = [(int(jj), float(dd)) for ii, jj, dd in zip(i, j, d) if ii == h_i_index]
    neigh_j = [(int(jj), float(dd)) for ii, jj, dd in zip(i, j, d) if ii == h_j_index]

    meta = {
        "occupied_o_indices": [int(o_i), int(o_j)],
        "o_indices": [int(o_i), int(o_j)],
        "h_indices": [int(h_i_index), int(h_j_index)],
        "oh_distance_A": float(oh_distance),
        "o_positions_A": [[float(x) for x in oi_pos], [float(x) for x in oj_pos]],
        "h_positions_A": [[float(x) for x in hi_pos], [float(x) for x in hj_pos]],
        "zmax_A": float(z_top),
        "z_freeze_A": float(z_freeze),
        "neighbor_cutoff_A": float(neighbor_cutoff),
        "neighbors_of_H": {
            str(h_i_index): [{"index": int(jj), "distance_A": float(dd)} for jj, dd in neigh_i],
            str(h_j_index): [{"index": int(jj), "distance_A": float(dd)} for jj, dd in neigh_j],
        },
        "num_atoms_total": int(len(slab)),
        "warnings": warnings,
    }

    return slab, meta


def main():
    ap = argparse.ArgumentParser(description="Generate 2-H pair structures for IrO2 Ising fitting.")
    ap.add_argument("--input", default="inputs/slab_clean_2x2.in", help="QE espresso-in slab file")
    ap.add_argument("--outputs", default="outputs", help="Root outputs dir")
    ap.add_argument("--center-o-index", type=int, default=20, help="For folder naming consistency (batch_centerOXX)")
    ap.add_argument("--oh-dist", type=float, default=1.0, help="O–H placement distance (Å)")
    ap.add_argument("--z-freeze", type=float, default=20.0, help="Freeze atoms below this z (Å)")
    ap.add_argument("--neighbor-cutoff", type=float, default=1.5, help="Neighbor cutoff (Å)")
    ap.add_argument("--surface-threshold", type=float, default=1.5, help="Surface O detection threshold (Å)")
    ap.add_argument("--pairs", default="", help="Optional: comma-separated pairs like '21-23,44-46'.")
    args = ap.parse_args()

    outdir = Path(args.outputs)
    outdir.mkdir(parents=True, exist_ok=True)

    # Read the clean slab once (same as your current setup script) and disable PBC for xTB
    slab0 = read(args.input, format="espresso-in")
    slab0.set_pbc((False, False, False))

    # Decide which pairs to generate
    if args.pairs.strip():
        pairs = []
        for token in args.pairs.split(","):
            token = token.strip()
            if not token:
                continue
            a, b = token.split("-")
            pairs.append((int(a), int(b)))
    else:
        pairs = DEFAULT_PAIRS

    batch_root = outdir / f"batch_centerO{args.center_o_index}" / "pairs"
    batch_root.mkdir(parents=True, exist_ok=True)

    manifest = {
        "input": str(args.input),
        "center_o_index": int(args.center_o_index),
        "pairs": [],
    }

    for (o_i, o_j) in pairs:
        tag = f"o{o_i}_o{o_j}"
        run_dir = batch_root / tag
        (run_dir / "structures").mkdir(parents=True, exist_ok=True)
        (run_dir / "results").mkdir(parents=True, exist_ok=True)

        slab, meta = prepare_pair_structure(
            slab0=slab0,
            o_i=o_i,
            o_j=o_j,
            oh_distance=args.oh_dist,
            z_freeze=args.z_freeze,
            neighbor_cutoff=args.neighbor_cutoff,
            surface_threshold=args.surface_threshold,
        )

        # File naming mirrors your existing convention but with both O indices in base_name
        base_name = f"slab_2H_o{o_i}_o{o_j}_ready"
        traj_path = run_dir / "structures" / f"{base_name}.traj"
        meta_path = run_dir / "results" / f"metadata_{base_name}.json"

        meta.update({
            "base_name": base_name,
            "structure_file": str(traj_path),
        })

        write(str(traj_path), slab)
        meta_path.write_text(json.dumps(meta, indent=2))

        manifest["pairs"].append({
            "o_i": int(o_i),
            "o_j": int(o_j),
            "base_name": base_name,
            "traj": str(traj_path),
            "meta": str(meta_path),
        })

        print(f"[ok] wrote {traj_path}")
        print(f"[ok] wrote {meta_path}")

    (batch_root / "pair_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"\nWrote manifest: {batch_root / 'pair_manifest.json'}")
    print("\nNext step: optimize each .traj with optimization_iro2_test.py")


if __name__ == "__main__":
    main()
