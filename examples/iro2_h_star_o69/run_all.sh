#!/bin/bash
set -euo pipefail

# ---- Helpers: run commands inside conda env if available, otherwise run directly
run_py () {
  if command -v conda >/dev/null 2>&1; then
    conda run -n iro2 python -u "$@"
  else
    python -u "$@"
  fi
}

run_sh () {
  if command -v conda >/dev/null 2>&1; then
    conda run -n iro2 bash "$@"
  else
    bash "$@"
  fi
}

echo "==> Canonical example: IrO2 + H* (o69)"

# --- Input expected from scripts/generate_states.py (your existing CI uses this path)
TRAJ="outputs/states/H_star/structures/slab_H_o69_ready.traj"

# --- Outputs
OPT_OUTDIR="outputs/states/H_star/results"
HAM_OUTDIR="outputs/hamiltonians/o69"

# --- Sanity checks
if [[ ! -f "$TRAJ" ]]; then
  echo "ERROR: Missing $TRAJ"
  echo "Hint: run state generation first, e.g.:"
  echo "  conda run -n iro2 python -u scripts/generate_states.py --state_set inputs/state_set.json --outputs outputs"
  exit 1
fi

mkdir -p "$OPT_OUTDIR" "$HAM_OUTDIR"

echo "==> Step 1: Geometry optimisation"
# Match your existing CI convention: --input/--output
run_py scripts/optimization_iro2_test.py --input "$TRAJ" --output "$OPT_OUTDIR"

# Locate optimised structure (be robust to naming differences)
OPT_TRAJ=""
if [[ -f "$OPT_OUTDIR/slab_H_o69_ready_opt.traj" ]]; then
  OPT_TRAJ="$OPT_OUTDIR/slab_H_o69_ready_opt.traj"
else
  # fallback: first traj in output dir
  OPT_TRAJ="$(ls -1 "$OPT_OUTDIR"/*.traj 2>/dev/null | head -n 1 || true)"
fi

if [[ -z "${OPT_TRAJ}" || ! -f "${OPT_TRAJ}" ]]; then
  echo "ERROR: No optimised .traj found in $OPT_OUTDIR"
  echo "Directory contents:"
  ls -la "$OPT_OUTDIR" || true
  exit 1
fi

echo "Optimised structure: $OPT_TRAJ"

echo "==> Step 2: Fermionic Hamiltonian (optional; requires pyscf)"
if run_py -c "import pyscf" >/dev/null 2>&1; then
  # These flags are the most common pattern; adjust if your script uses different names.
  run_py scripts/build_fermionic_hamiltonian_pyscf.py --structure "$OPT_TRAJ" --outdir "$HAM_OUTDIR"
else
  echo "Skipping fermionic Hamiltonian: pyscf not available in env 'iro2'"
fi

echo "==> Step 3: Export qubit Hamiltonian JSON (if active-space fermionic Hamiltonian exists)"
FERM_ACTIVE="$HAM_OUTDIR/fermionic_active_space.json"
QUBIT_JSON="$HAM_OUTDIR/qubit_hamiltonian.json"

if [[ -f "$FERM_ACTIVE" ]]; then
  run_py scripts/export_qubit_hamiltonian_json.py --input "$FERM_ACTIVE" --output "$QUBIT_JSON"
else
  echo "Skipping qubit export: $FERM_ACTIVE not found"
fi

echo "==> Example workflow complete."
