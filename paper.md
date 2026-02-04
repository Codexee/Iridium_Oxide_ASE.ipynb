# A Reproducible Workflow for Extracting Quantum Hamiltonians from Surface–Adsorbate Models

**Manisha Malhotra**¹, **Neil Ramarapu**², **Fabio Rinaldi**³

¹ *Westward Academy, USA*
² *Thomas Jefferson High School, Virginia, USA*
³ *Google, Dublin, Ireland*

January 2026

## Summary

Bridging the gap between realistic surface catalysis models from density functional theory (DFT) and the input requirements of quantum algorithms is a non-trivial task. Surface DFT calculations typically model extended slabs and yield total energies or reaction energetics, whereas quantum algorithms (such as variational quantum eigensolvers) require a many-body Hamiltonian (e.g. a list of fermionic or qubit operators) as input.

We present an open-source, reproducible software workflow that takes an optimized DFT slab model of a catalyst surface and systematically produces a few-body Hamiltonian suitable for quantum simulation. Starting from a DFT-optimized slab geometry, the workflow extracts a localized cluster around the active site, identifies an appropriate set of active orbitals, systematically produces a few-body Hamiltonian suitable for quantum simulation, with Hamiltonian construction provided as an optional stage of the workflow. This end-to-end pipeline enables researchers to bridge surface science and quantum computing in a transparent and automated manner.

We demonstrate the workflow on a representative electrochemical system: a hydrogen-based adsorbate (H* or OH*) on an IrO₂ catalyst surface. For this system, our software produces a 14-qubit Hamiltonian (derived from a 7-orbital active space) that can be directly used in quantum algorithms such as the Variational Quantum Eigensolver (VQE [@VQE]).

## Statement of Need

The hydrogen evolution reaction (HER) is a key process in electrochemistry (2H⁺ + 2e⁻ → H₂) and its efficiency is greatly influenced by the interaction between hydrogen and catalyst surface sites. While DFT has been the workhorse for studying such catalytic systems, it often struggles when confronted with strongly correlated electrons or multiple spin-state character that can occur in transition metal compounds.

In particular, iridium-based catalysts present a challenge: Ir is a heavy transition metal where relativistic effects and variable oxidation states can render standard DFT approximate or unreliable in capturing certain electronic structure features (such as spin-state changes upon adsorption). To our knowledge, no prior studies have applied quantum algorithms to model HER on Ir-based catalysts, highlighting a methodological gap.

Quantum computing offers a promising avenue to address this challenge by enabling explicitly correlated, multireference treatments of active-site electrons. Algorithms like the Variational Quantum Eigensolver (VQE [@VQE]) can efficiently target ground-state energies of strongly correlated systems using quantum hardware [@Qiskit; @Superstaq], and have already demonstrated accuracy comparable to classical methods for challenging molecular systems.

However, a major obstacle remains: how to obtain the required fermionic Hamiltonian from a realistic catalyst model. Existing surface science workflows typically output energies, charge distributions, or density-of-states information, but do not yield second-quantized Hamiltonians or an explicit orbital basis suitable for quantum algorithms. There is currently no standard pipeline to go from a periodic slab DFT model to a localized active-space Hamiltonian.

Our software addresses this unmet need by providing a reproducible pipeline that takes an optimized slab model as input and produces a Hamiltonian ready for quantum simulation. This capability is especially important for systems like IrO₂-catalyzed HER, where the surface can cycle through oxide and hydroxide states involving multiple Ir oxidation states. By facilitating the construction of active-space Hamiltonians for these challenging catalytic sites, our workflow opens the door to applying resource-aware quantum algorithms to heterogeneous catalysis.

## Software Description


### Pipeline Overview Figure

![Overview of the workflow from DFT slab to qubit Hamiltonian. The pipeline shows slab optimization, cluster extraction, active-space selection, fermionic Hamiltonian construction, and qubit mapping.](figures/pipeline_overview.png)

### Workflow Overview

The workflow proceeds through a sequence of transformations from a realistic surface model to a minimal quantum simulation model:

1. **Slab optimization (DFT):** A periodic DFT optimization of the catalyst surface with the adsorbate of interest is performed externally (e.g. using Quantum ESPRESSO [@QE] or VASP) to obtain a relaxed slab geometry.

2. **Cluster extraction (ASE):** Using the Atomic Simulation Environment (ASE) [@ASE], a finite cluster containing the active site is extracted from the slab. The cluster captures the local chemical environment while reducing system size. Peripheral bonds may be capped with hydrogens.

3. **Orbital analysis and active-space selection:** A semiempirical tight-binding calculation (GFN-xTB [@xTB]) is used to obtain approximate orbital energies. Frontier orbitals relevant to adsorption chemistry are selected to define an active space.

4. **Fermionic Hamiltonian construction (PySCF [@PySCF]):** A quantum chemistry calculation is performed on the cluster to obtain one- and two-electron integrals for the active orbitals. These integrals define a second-quantized fermionic Hamiltonian.

5. **Qubit mapping (OpenFermion [@OpenFermion]):** The fermionic Hamiltonian is mapped to a qubit Hamiltonian (e.g. via the Jordan–Wigner transformation) and exported in a machine-readable format suitable for quantum algorithms.

Each stage of the pipeline reduces complexity in a controlled and reproducible manner, transforming an extended slab into a few-qubit Hamiltonian while preserving the essential physics of the surface–adsorbate interaction.

### Implementation Details

The software is implemented in Python and builds on open-source libraries including ASE, xTB, PySCF [@PySCF], and OpenFermion [@OpenFermion]. ASE is used for structure handling and cluster extraction, while xTB provides rapid orbital screening. PySCF [@PySCF] is employed for higher-fidelity electronic structure calculations and integral generation within the selected active space.

OpenFermion [@OpenFermion] is used to perform fermion-to-qubit mappings such as Jordan–Wigner or Bravyi–Kitaev. The resulting qubit Hamiltonians are stored in JSON format together with metadata to ensure clarity and reproducibility.

The repository includes continuous integration tests using GitHub Actions to ensure that updates to the code do not break the workflow. A Jupyter notebook example demonstrates the full pipeline on an IrO₂ surface with an adsorbate, producing a reference Hamiltonian that is automatically checked for consistency. To ensure reasonable execution times for continuous integration and example usage, the most computationally expensive stages of the workflow—specifically the construction and qubit mapping of the full fermionic Hamiltonian—are provided as optional steps. The default example workflow executes slab processing, cluster extraction, and orbital analysis, while Hamiltonian generation can be enabled by the user when sufficient computational resources are available.

## Application

*(Relevant prior surface-science and catalysis studies are cited where appropriate to provide context for the chosen adsorption site and system, independent of the software focus of this work.)*

The choice of adsorption site and surface chemistry is informed by prior computational studies of IrO₂ surfaces under electrochemical conditions [@ReshmaIrO2]. More broadly, this work is motivated by recent efforts to connect materials modeling with quantum computational workflows for realistic condensed-matter systems [@InspiredWorkflow].

The workflow was applied to a hydroxyl species adsorbed on a rutile IrO₂(110) surface at the o69 oxygen-bridge site. A finite cluster of 38 atoms was extracted from a periodic slab and capped with hydrogens. Orbital analysis identified the OH bonding orbital, nearby Ir d orbitals, and bridging O p orbitals as the most chemically relevant.

An active space of 7 spatial orbitals (14 spin orbitals) was selected, leading to a 14-qubit Hamiltonian. One- and two-electron integrals were computed using PySCF [@PySCF] with a minimal basis set, and the Hamiltonian was mapped to qubits using the Jordan–Wigner transformation.

The resulting Hamiltonian contains on the order of 10⁴ Pauli terms and was tested using VQE [@VQE] on both simulators and available quantum hardware. The measured ground-state energies were consistent with classical diagonalization within expected hardware error margins, confirming the correctness and usability of the generated Hamiltonian.

## Availability

The software is openly available on GitHub at **[https://github.com/Codexee/Iridium_Oxide_ASE](https://github.com/Codexee/Iridium_Oxide_ASE)** under the MIT License.

The repository includes preconfigured examples that demonstrate the full workflow logic without requiring long-running quantum chemistry calculations. Complete Hamiltonian generation is reproducible using the same code paths and parameters and is documented in the repository, but is not executed by default in automated tests.

An archived version of the repository will be created and assigned a DOI upon successful completion of the JOSS review, in accordance with JOSS submission guidelines.

## Reproducibility

The repository contains the Python source code, documentation, example notebooks, and reference outputs required to reproduce the results presented in this paper. Continuous integration ensures that the workflow remains reproducible across updates.

The archived release will provide a static snapshot of the codebase used to generate the Hamiltonians reported here. All parameters and choices in the workflow are deterministic or explicitly controlled, ensuring that the example results can be reproduced exactly given the specified software versions.

## AI Disclosure

This manuscript and accompanying software documentation were prepared with the assistance of an AI-based language model. The AI system was used to support tasks such as Markdown conversion, structural editing for JOSS compliance, and language refinement. All scientific content, technical decisions, interpretations, and conclusions were reviewed, validated, and approved by the authors, who take full responsibility for the work.

## Future Work

Future developments include automated orbital freezing and Hamiltonian reduction techniques to further lower qubit requirements, as well as tighter integration with quantum algorithms such as VQE [@VQE] and QITE. We also plan to extend the workflow to additional catalytic systems, charged clusters, and more realistic electrochemical environments.

By continuing to refine and expand the pipeline, we aim to establish a standard toolkit for quantum computational catalysis, enabling routine application of quantum algorithms to complex surface chemistry problems.

## References

References are provided in a separate BibTeX file (`paper.bib`) and are cited in the text using standard Pandoc/JOSS citation syntax (e.g. `[@PySCF]`).
