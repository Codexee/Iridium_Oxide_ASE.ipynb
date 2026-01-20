#!/usr/bin/env python3
"""
Fit an occupancy-Ising model with multi-shell pair interactions to relaxed energies,
optionally weighted by fmax, and export as Pauli-Z coefficients.

Input dataset format (JSON):
{
  "candidate_site_atom_indices": [12, 15, 18, ...],   # length Nsites, indices in an ASE structure
  "reference_structure": "path/to/a.traj",            # used to get site coordinates + cell
  "samples": [
    {"name":"slab_H_o21_ready", "occupied_site_ids":[0,3,5,7, ...], "energy": -10386.91, "fmax":0.0498},
    ...
  ]
}

- occupied_site_ids are indices into candidate_site_atom_indices (0..Nsites-1)
- energy should be a consistent quantity (total energy or adsorption energy), in eV.
"""

from __future__ import annotations
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional

import numpy as np

# Optional dependency: ASE for reading structure and using MIC distances
try:
    from ase.io import read as ase_read
except Exception as e:
    ase_read = None


@dataclass(frozen=True)
class FitConfig:
    shells: List[float]          # shell edges in Angstrom, e.g. [3.0, 5.0, 7.0]
    ridge: float = 1e-8          # small ridge to stabilize
    sigma0: float = 1e-3         # eV baseline for weights
    alpha: float = 0.05          # eV/(eV/Å) scale for weights
    fmax_filter: Optional[float] = None  # e.g. 0.06 to drop unconverged points


def _mic_distance(cell: np.ndarray, pbc: np.ndarray, r_i: np.ndarray, r_j: np.ndarray) -> float:
    """
    Minimal image distance. If ASE is available, you'd normally use ase.geometry, but
    this simple implementation is OK for orthorhombic-ish cells.
    """
    # For general cells, a robust MIC needs fractional coords + wrap.
    # We'll do a conservative approach: if cell is not usable, fall back to direct distance.
    if cell is None or not np.any(pbc):
        return float(np.linalg.norm(r_i - r_j))

    try:
        # Convert to fractional
        inv_cell = np.linalg.inv(cell.T)
        fi = inv_cell @ r_i
        fj = inv_cell @ r_j
        df = fi - fj
        # wrap in periodic directions
        for k in range(3):
            if pbc[k]:
                df[k] -= np.round(df[k])
        dr = cell.T @ df
        return float(np.linalg.norm(dr))
    except Exception:
        return float(np.linalg.norm(r_i - r_j))


def build_shell_pairs(
    site_positions: np.ndarray,
    cell: Optional[np.ndarray],
    pbc: Optional[np.ndarray],
    shells: List[float],
) -> Tuple[List[Tuple[int, int]], List[int]]:
    """
    Create (i,j) pairs and assign each to a shell index based on distance.
    Returns:
      pairs: list of (i,j), i<j
      shell_id: list of same length, integer in [0..nshell-1]
    """
    N = site_positions.shape[0]
    edges = list(shells)
    ns = len(edges)

    pairs: List[Tuple[int,int]] = []
    shell_ids: List[int] = []

    for i in range(N):
        for j in range(i+1, N):
            d = _mic_distance(cell, pbc, site_positions[i], site_positions[j])
            # find first shell edge that contains d
            sid = None
            for s, edge in enumerate(edges):
                if d <= edge:
                    sid = s
                    break
            if sid is None:
                continue  # outside max shell cutoff
            pairs.append((i, j))
            shell_ids.append(sid)

    return pairs, shell_ids


def build_design_matrix_shells(
    occ: np.ndarray,                 # K x N (0/1)
    pairs: List[Tuple[int,int]],
    pair_shell_ids: List[int],
    nshell: int,
) -> np.ndarray:
    """
    Columns: [1 | singles(n_i) | pair_shell_0_sum | ... | pair_shell_{ns-1}_sum]
    where pair_shell_s_sum = sum_{(i,j) in shell s} n_i n_j
    """
    K, N = occ.shape
    X = np.zeros((K, 1 + N + nshell), dtype=float)
    X[:, 0] = 1.0
    X[:, 1:1+N] = occ

    # accumulate shell sums
    for idx, (i, j) in enumerate(pairs):
        s = pair_shell_ids[idx]
        X[:, 1+N+s] += occ[:, i] * occ[:, j]

    return X


def weighted_ridge_fit(X: np.ndarray, y: np.ndarray, w: np.ndarray, ridge: float) -> np.ndarray:
    """
    Solve (X^T W X + ridge I) beta = X^T W y
    """
    y = y.reshape(-1, 1)
    W = np.diag(w)
    XtW = X.T @ W
    A = XtW @ X + ridge * np.eye(X.shape[1])
    b = XtW @ y
    beta = np.linalg.solve(A, b).reshape(-1)
    return beta


def occ_to_pauli_z(E0: float, h: np.ndarray, J_pairs: Dict[Tuple[int,int], float]) -> Dict[str, Any]:
    """
    Convert occupancy-Ising parameters to Pauli-Z coefficients.

    Returns dict:
      {
        "constant": C,
        "Z": {i: a_i},
        "ZZ": {"i,j": b_ij}
      }
    """
    N = len(h)
    a = np.zeros(N, dtype=float)
    b: Dict[str, float] = {}

    # ZZ terms
    for (i, j), Jij in J_pairs.items():
        bij = Jij / 4.0
        b[f"{i},{j}"] = bij
        a[i] += -Jij / 4.0
        a[j] += -Jij / 4.0

    # Z terms from singles
    a += -h / 2.0

    # Constant
    C = float(E0 + np.sum(h / 2.0) + np.sum([Jij / 4.0 for Jij in J_pairs.values()]))

    return {"constant": C, "Z": {int(i): float(a[i]) for i in range(N)}, "ZZ": b}


def main(
    dataset_json: str,
    out_json: str = "ising_fit_out.json",
    shells: str = "3.0,5.0,7.0",
    ridge: float = 1e-8,
    sigma0: float = 1e-3,
    alpha: float = 0.05,
    fmax_filter: Optional[float] = None,
):
    dataset_path = Path(dataset_json)
    data = json.loads(dataset_path.read_text())

    cand_atom_indices = data["candidate_site_atom_indices"]
    ref_struct = data["reference_structure"]
    samples = data["samples"]

    if ase_read is None:
        raise RuntimeError("ASE not available. Please `pip install ase` in your environment.")

    atoms = ase_read(ref_struct)
    site_positions = np.array([atoms[idx].position for idx in cand_atom_indices], dtype=float)
    cell = np.array(atoms.cell, dtype=float) if atoms.cell is not None else None
    pbc = np.array(atoms.pbc, dtype=bool) if atoms.pbc is not None else np.array([False, False, False], dtype=bool)

    shell_edges = [float(x.strip()) for x in shells.split(",") if x.strip()]
    pairs, shell_ids = build_shell_pairs(site_positions, cell, pbc, shell_edges)

    N = len(cand_atom_indices)
    K = len(samples)

    occ = np.zeros((K, N), dtype=int)
    y = np.zeros(K, dtype=float)
    fmax = np.zeros(K, dtype=float)
    names: List[str] = []

    for k, s in enumerate(samples):
        names.append(s.get("name", f"sample_{k}"))
        y[k] = float(s["energy"])
        fmax[k] = float(s.get("fmax", 0.0))
        for sid in s["occupied_site_ids"]:
            occ[k, int(sid)] = 1

    # Optional filtering
    keep = np.ones(K, dtype=bool)
    if fmax_filter is not None:
        keep &= (fmax <= float(fmax_filter))

    occ_f = occ[keep]
    y_f = y[keep]
    fmax_f = fmax[keep]
    names_f = [n for n, kk in zip(names, keep) if kk]

    # Weights from fmax (or all ones)
    w = 1.0 / (sigma0 + alpha * fmax_f) ** 2

    X = build_design_matrix_shells(occ_f, pairs, shell_ids, nshell=len(shell_edges))
    beta = weighted_ridge_fit(X, y_f, w, ridge=ridge)

    E0 = float(beta[0])
    h = beta[1:1+N].copy()
    J_shell = beta[1+N:]  # length nshell

    # Expand shell Js into per-pair J_ij (all pairs in same shell share J)
    J_pairs: Dict[Tuple[int,int], float] = {}
    for (i, j), sid in zip(pairs, shell_ids):
        J_pairs[(i, j)] = float(J_shell[sid])

    pauli = occ_to_pauli_z(E0, h, J_pairs)

    out = {
        "fit": {
            "E0": E0,
            "h": [float(x) for x in h],
            "J_shell_edges_A": shell_edges,
            "J_shell": [float(x) for x in J_shell],
            "n_sites": N,
            "n_samples_used": int(len(y_f)),
            "weights": {"sigma0": sigma0, "alpha": alpha, "fmax_filter": fmax_filter},
            "feature_counts": {"pairs_total": len(pairs), "shells": len(shell_edges)},
        },
        "pauli_Z": pauli,
        "meta": {
            "dataset": str(dataset_path),
            "reference_structure": ref_struct,
            "candidate_site_atom_indices": cand_atom_indices,
            "samples_used": names_f,
        }
    }

    Path(out_json).write_text(json.dumps(out, indent=2))
    print(f"Wrote: {out_json}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, help="Path to dataset JSON (see docstring).")
    ap.add_argument("--out", default="ising_fit_out.json")
    ap.add_argument("--shells", default="3.0,5.0,7.0")
    ap.add_argument("--ridge", type=float, default=1e-8)
    ap.add_argument("--sigma0", type=float, default=1e-3)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--fmax_filter", type=float, default=None)
    args = ap.parse_args()

    main(
        dataset_json=args.dataset,
        out_json=args.out,
        shells=args.shells,
        ridge=args.ridge,
        sigma0=args.sigma0,
        alpha=args.alpha,
        fmax_filter=args.fmax_filter,
    )
