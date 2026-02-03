#!/bin/bash
set -e

echo "Building slab and adsorbate..."
python scripts/build_slab_and_adsorbate.py

echo "Optimising geometry..."
python scripts/optimize_structure.py outputs/states/H_star/o69/slab_H_o69_ready.traj

echo "Extracting Hamiltonian..."
python scripts/extract_hamiltonian.py \
  outputs/results/H_star/o69/slab_H_o69_ready_opt.traj

echo "Running quantum simulation..."
python scripts/run_vqe_simulation.py \
  outputs/hamiltonians/o69/fermionic_active_space.json

echo "Example workflow complete."
