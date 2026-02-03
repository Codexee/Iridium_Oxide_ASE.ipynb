Example: Hydrogen adsorption on an IrO₂ surface

This example demonstrates the full classical-to-quantum workflow implemented in this repository, using hydrogen adsorption on an IrO₂ surface as a representative test case.

The goal of this example is not to obtain chemically converged adsorption energies, but to illustrate a reproducible, end-to-end pipeline from atomistic modelling to quantum-ready Hamiltonians.

Overview of the workflow

  The example proceeds through the following steps:

    1. Construction of an IrO₂ surface slab with a hydrogen adsorbate

    2. Geometry optimisation using a semi-empirical method

    3. Extraction of a second-quantized electronic Hamiltonian

    4. Active space reduction informed by chemical locality

    5. Mapping to qubits and solution using a simulated variational quantum eigensolver (VQE)

  All steps can be executed locally and do not require access to quantum hardware.

Requirements

  This example assumes that the main package dependencies have been installed. In particular:

    - ASE

    - NumPy

    - Qiskit (Aer simulator)

    - xTB (for geometry optimisation)

  See the main repository README for installation instructions.

Running the full example

  From the root of the repository, run:

    bash examples/iro2_h_star_o69/run_all.sh

  This script executes the full workflow sequentially. Intermediate and final outputs are written to the outputs/ directory.

Step-by-step description
  1. Slab and adsorbate construction

    An IrO₂ slab is constructed and a hydrogen atom is placed at a selected surface oxygen site (o69).

    Script:

    python scripts/build_slab_and_adsorbate.py


    Output:

    outputs/states/H_star/o69/slab_H_o69_ready.traj

  2. Geometry optimisation

    The adsorbed structure is relaxed using a semi-empirical method to obtain a low-cost reference geometry.

  Script:

  python scripts/optimize_structure.py outputs/states/H_star/o69/slab_H_o69_ready.traj


  Outputs:

  outputs/results/H_star/o69/
   ├── slab_H_o69_ready_opt.traj
   ├── slab_H_o69_ready_opt.log
   └── slab_H_o69_ready_results.json


  The results JSON includes convergence metadata and total energies.

3. Hamiltonian extraction and active space selection

  From the optimised structure, a second-quantized electronic Hamiltonian is generated. An active space is selected based on chemically local orbitals near the adsorption site.

  Script:

    python scripts/extract_hamiltonian.py \
    outputs/results/H_star/o69/slab_H_o69_ready_opt.traj


  Outputs:

  outputs/hamiltonians/o69/
   ├── fermionic_full.json
   ├── fermionic_active_space.json
   └── fermionic_active_space.npz

4. Qubit mapping and quantum simulation

  The reduced Hamiltonian is mapped to a qubit operator and solved using a variational quantum eigensolver (VQE) on a noiseless simulator.

Script:

  python scripts/run_vqe_simulation.py \
    outputs/hamiltonians/o69/fermionic_active_space.json


Outputs:

  outputs/quantum/o69/
   ├── vqe_energy.json
   └── circuit_metadata.json


  This step demonstrates quantum compatibility and resource scaling, rather than chemically accurate energies.

5. (Optional) Hardware execution

  The same workflow can be extended to hardware backends such as Superstaq. This example includes a dry-run script illustrating hardware dispatch without consuming credits.

  python scripts/run_vqe_superstaq.py \
    --dry-run \
    outputs/hamiltonians/o69/fermionic_active_space.json

Notes on scope

  This example is intended as a demonstration of software capability.

  Quantum simulations are illustrative and not intended to provide quantitatively converged adsorption energies.

  The workflow is extensible to other materials systems and adsorbates.

  Reproducing or extending the example

  Users can adapt this example by:

  selecting different surface sites, modifying the active space selection strategy or substituting alternative quantum algorithms.
