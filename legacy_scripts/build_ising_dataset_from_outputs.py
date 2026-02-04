#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Any, List, Tuple

def resolve_path(p: str, root: Path) -> str:
    """
    Make paths robust to artifact unpack location.
    - If p exists as-is, return it.
    - If p starts with 'outputs/', try stripping it.
    - Otherwise search for the filename under root.
    """
    if not p:
        return p

    p_path = Path(p)
    if p_path.exists():
        return str(p_path)

    # Common case: results JSON stored paths under outputs/..., but artifact unpacked into repo root
    if p.startswith("outputs/"):
        stripped = Path(p[len("outputs/"):])
        if stripped.exists():
            return str(stripped)

    # Try relative to root explicitly
    rel = root / p
    if rel.exists():
        return str(rel)

    # Last resort: search by filename
    fname = p_path.name
    hits = list(root.glob(f"**/{fname}"))
    if hits:
        # Prefer the shortest path (often the closest match)
        hits_sorted = sorted(hits, key=lambda x: len(str(x)))
        return str(hits_sorted[0])

    return p  # leave as-is; downstream will error with a clear missing file


def load_json(p: Path) -> Dict[str, Any]:
    return json.loads(p.read_text())


def pick_ref_structure(result_json: Dict[str, Any]) -> str:
    # Prefer an explicit structure file if present; fall back to final_traj.
    for k in ("structure_file", "input_structure", "final_traj"):
        v = result_json.get(k)
        if isinstance(v, str) and v:
            return v
    return ""


def occupied_site_ids_from_meta(meta: Dict[str, Any], site_index_map: Dict[int, int]) -> List[int]:
    """
    Returns occupied_site_ids as indices into candidate_site_atom_indices.
    Supports:
      - meta["occupied_o_indices"] as list[int] (multi-occupancy)
      - meta["target_o_index"] as int (single-occupancy)
      - meta["bits"] as string like "00100000" (single-occupancy within K)
    """
    if "occupied_o_indices" in meta and isinstance(meta["occupied_o_indices"], list):
        out = []
        for o_idx in meta["occupied_o_indices"]:
            o_idx = int(o_idx)
            if o_idx not in site_index_map:
                raise KeyError(f"occupied_o_indices includes O[{o_idx}] not in candidate sites")
            out.append(site_index_map[o_idx])
        return sorted(set(out))

    if "target_o_index" in meta:
        o_idx = int(meta["target_o_index"])
        if o_idx not in site_index_map:
            raise KeyError(f"target_o_index O[{o_idx}] not in candidate sites")
        return [site_index_map[o_idx]]

    if "bits" in meta and isinstance(meta["bits"], str) and "1" in meta["bits"]:
        hot = meta["bits"].index("1")
        # bits is an index within K nearest sites; we assume site_index_map was built from those
        return [hot]

    raise KeyError("Could not determine occupied site(s): expected occupied_o_indices, target_o_index, or bits")


def main(outputs_dir: str, out_path: str):
    outdir = Path(outputs_dir)
    if not outdir.exists():
        raise FileNotFoundError(f"outputs_dir not found: {outdir}")

    # Collect metadata and results JSONs
    meta_files = sorted(outdir.glob("**/results/metadata_*.json"))
    result_files = sorted(outdir.glob("**/results/*_results.json"))

    if not meta_files:
        raise FileNotFoundError(f"No metadata_*.json found under {outdir}/**/results/")
    if not result_files:
        raise FileNotFoundError(f"No *_results.json found under {outdir}/**/results/")

    # Build candidate sites as all target_o_index values found in metadata files
    # (This matches your current batch generation: K nearest oxygen sites.)
    target_o_indices: List[int] = []
    metas: List[Tuple[Path, Dict[str, Any]]] = []
    for mf in meta_files:
        m = load_json(mf)
        metas.append((mf, m))
        if "target_o_index" in m:
            target_o_indices.append(int(m["target_o_index"]))

    if not target_o_indices:
        raise KeyError("No target_o_index found in metadata files. Cannot build candidate site list.")

    candidate_site_atom_indices = sorted(set(target_o_indices))
    site_index_map = {atom_idx: i for i, atom_idx in enumerate(candidate_site_atom_indices)}

    # Index results by base_name (your optimization writes base_name + *_results.json)
    results_by_base: Dict[str, Dict[str, Any]] = {}
    for rf in result_files:
        r = load_json(rf)
        base = r.get("base_name")
        if isinstance(base, str) and base:
            results_by_base[base] = r

    samples = []
    ref_structure = ""

    # Pair each metadata file to a result using the target site naming convention: slab_H_o{idx}_ready
    for mf, meta in metas:
        if "target_o_index" not in meta:
            continue
        o_idx = int(meta["target_o_index"])
        base_name = f"slab_H_o{o_idx}_ready"

        r = results_by_base.get(base_name)
        if r is None:
            # Some pipelines store results in a different folder/name; try to locate by filename stem
            # If no match, skip
            continue

        occ_ids = occupied_site_ids_from_meta(meta, site_index_map)

        energy = float(r["e_final"])
        fmax = float(r.get("fmax_final", 0.0))

        if not ref_structure:
            raw_ref = pick_ref_structure(r)
            ref_structure = resolve_path(raw_ref, outdir)


        samples.append(
            {
                "name": base_name,
                "occupied_site_ids": occ_ids,
                "energy": energy,
                "fmax": fmax,
            }
        )

    if not samples:
        raise RuntimeError(
            "No samples built. Check that metadata target_o_index names match results base_name."
        )

    dataset = {
        "candidate_site_atom_indices": candidate_site_atom_indices,
        "reference_structure": ref_structure,
        "samples": samples,
    }

    outp = Path(out_path)
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(dataset, indent=2))
    print(f"Wrote dataset: {outp}")
    print(f"  n_sites:   {len(candidate_site_atom_indices)}")
    print(f"  n_samples: {len(samples)}")
    print(f"  ref:       {ref_structure}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--outputs_dir", default="outputs", help="Root outputs dir (downloaded artifact).")
    ap.add_argument("--out", default="outputs/ising_dataset.json", help="Where to write the dataset JSON.")
    args = ap.parse_args()
    main(args.outputs_dir, args.out)
