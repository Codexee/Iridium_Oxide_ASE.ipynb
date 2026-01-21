# Iridium Oxide ASE

This repository contains an open-source, reproducible workflow for generating and analysing
surface-bound intermediates on IrO₂ electrocatalysts using the Atomic Simulation Environment (ASE).

The codebase is designed to support benchmarking and hypothesis generation for operando
synchrotron studies (e.g. AP-XPS / AP-XAS) and to interface with emerging hybrid
classical–quantum simulation methodologies.

---

## Repository layout

- `scripts/`  
  Python scripts for slab preparation, adsorption-site generation, geometry optimisation,
  and analysis.

- `inputs/`  
  Input files and reference structures used by the scripts (including slab geometries).

- `tests/`  
  Automated tests executed via GitHub Actions to validate core functionality.

---

## Reproducibility note

Initial IrO₂ slab construction and relaxation were performed using Quantum Espresso prior
to ASE-based analysis.

The full analysis workflow (slab preparation, adsorption-site generation, and optimisation
scripts) used to generate the benchmark data reported in the accompanying technical report
is currently under active development on a dedicated development branch.

The `main` branch contains stable examples and documentation; the development branch
contains the complete pipeline used to generate the reported benchmark dataset.

---

## Running locally

To run example scripts locally:

```bash
python scripts/iro2_test_1.py
