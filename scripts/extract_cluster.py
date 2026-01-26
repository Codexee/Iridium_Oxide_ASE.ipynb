#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from ase.io import read, write


def main():
    ap = argparse.ArgumentParser(description="Extract a local cluster around an adsorption site from an optimized slab.")
    ap.add_argument("--traj", required=True, help="Input optimized .traj (e.g. slab_H_o69_ready_final.traj)")
    ap.add_argument("--center_atom_index", type=int, default=None,
                    help="Atom index in the slab to center on (e.g. O site index). If omitted, uses H atom.")
    ap.add_argument("--cutoff", type=float, default=6.0, help="Radius cutoff in Angstrom")
    ap.add_argument("--outdir", default="outputs/clusters", help="Output directory")
    ap.add_argument("--tag", default="o69", help="Tag name for output files")
    args = ap.parse_args()

    atoms = read(args.traj)

    if args.center_atom_index is None:
        # find H atom(s)
        H_indices = [i for i, a in enumerate(atoms) if a.symbol == "H"]
        if not H_indices:
            raise SystemExit("No H atom found. Provide --center_atom_index.")
        center_i = H_indices[0]
    else:
        center_i = args.center_atom_index

    center = atoms.positions[center_i]
    d = np.linalg.norm(atoms.positions - center, axis=1)
    keep = np.where(d <= args.cutoff)[0].tolist()

    cluster = atoms[keep]
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    out_traj = outdir / f"{args.tag}_cluster.traj"
    out_xyz = outdir / f"{args.tag}_cluster.xyz"
    write(out_traj, cluster)
    write(out_xyz, cluster)

    print(f"[ok] kept {len(cluster)} atoms within {args.cutoff:.2f} Å of index {center_i}")
    print(f"[ok] wrote {out_traj}")
    print(f"[ok] wrote {out_xyz}")


if __name__ == "__main__":
    main()
