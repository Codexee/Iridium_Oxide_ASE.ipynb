import os
import json
import warnings
import numpy as np
from pathlib import Path

from qiskit.quantum_info import SparsePauliOp
from qiskit.circuit.library import TwoLocal
from qiskit_algorithms.minimum_eigensolvers import VQE
from qiskit_algorithms.optimizers import COBYLA

# load JW JSON

json_path = Path("inputs/qubit_hamiltonian_jw.json")

def load_jw_json(json_path: list[str]) -> tuple[dict, str]:
    last_err = None
    for p in json_path:
        try:
            with open(p, "r") as f:
                return json.load(f), p
        except Exception as e:
            last_err = e
    raise FileNotFoundError(
        "Could not open qubit_hamiltonian_jw.json. Tried:\n  - "
        + "\n  - ".join(json_path)
        + f"\nLast error: {last_err}"
    )

def jw_to_sparsepauliop(jw_obj: dict) -> SparsePauliOp:
    """
    JW JSON format:
      jw_obj["terms"] is a list of terms.
      Each term has:
        term["paulis"] = list of [q, 'X'/'Y'/'Z'] entries
        term["coeff_real"], term["coeff_imag"]
    """
    n = int(jw_obj["n_qubits"])
    labels: list[str] = []
    coeffs: list[complex] = []

    for term in jw_obj["terms"]:
        label = ["I"] * n
        for q, p in term["paulis"]:
            q = int(q)
            if q < 0 or q >= n:
                raise ValueError(f"Qubit index {q} out of range for n_qubits={n}")
            if p not in ("I", "X", "Y", "Z"):
                raise ValueError(f"Invalid Pauli '{p}' in term {term}")
            label[n - 1 - q] = p
        labels.append("".join(label))
        coeffs.append(complex(float(term["coeff_real"]), float(term["coeff_imag"])))

    return SparsePauliOp.from_list(list(zip(labels, coeffs)))

# vqe

def build_ansatz(num_qubits: int) -> TwoLocal:
    # TwoLocal is deprecated in Qiskit >= 2.1
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=DeprecationWarning)
        ansatz = TwoLocal(
            num_qubits,
            rotation_blocks=["ry", "rz"],
            entanglement_blocks="cx",
            entanglement="linear",
            reps=2,
        )
    return ansatz

def run_vqe_local_aer(H: SparsePauliOp) -> float:
    from qiskit_aer.primitives import EstimatorV2 as AerEstimatorV2

    ansatz = build_ansatz(H.num_qubits)
    optimizer = COBYLA(maxiter=300)

    estimator = AerEstimatorV2()
    vqe = VQE(estimator=estimator, ansatz=ansatz, optimizer=optimizer)
    res = vqe.compute_minimum_eigenvalue(H)

    e0 = float(np.real(res.eigenvalue))
    return e0

def run_vqe_superstaq_if_backendv2(H: SparsePauliOp, target: str = "ibmq_fez") -> float | None:
    """
    Attempts to run VQE on a Superstaq backend.
    This requires the backend object to be a Qiskit BackendV2 so we can wrap it in BackendEstimatorV2
    (which implements the V2 estimator interface required by qiskit-algorithms VQE).
    """
    import qiskit_superstaq as qss
    from qiskit.providers import BackendV2
    from qiskit.primitives import BackendEstimatorV2

    api_key = ".."
    if not api_key:
        raise RuntimeError("SUPERSTAQ_API_KEY environment variable is not set.")

    provider = qss.SuperstaqProvider(api_key=api_key)
    backend = provider.get_backend(target)

    if not isinstance(backend, BackendV2):
        print(f"Superstaq backend '{target}' is type {type(backend)} (not BackendV2). Skipping Superstaq VQE.")
        return None

    ansatz = build_ansatz(H.num_qubits)
    optimizer = COBYLA(maxiter=300)

    estimator = BackendEstimatorV2(backend=backend, options={"shots": 20000})
    vqe = VQE(estimator=estimator, ansatz=ansatz, optimizer=optimizer)
    res = vqe.compute_minimum_eigenvalue(H)

    e0 = float(np.real(res.eigenvalue))
    return e0

def main():
    jw, path_used = load_jw_json(
        json_path)
    H = jw_to_sparsepauliop(jw)

    print(f"Loaded JW Hamiltonian from: {path_used}")
    print(f"n_qubits = {H.num_qubits}, n_terms = {len(H)}")

    e_local = run_vqe_local_aer(H)
    print("VQE minimum eigenvalue (local Aer):", e_local)

    try:
        e_super = run_vqe_superstaq_if_backendv2(H, target="ibmq_fez")
        if e_super is not None:
            print("VQE minimum eigenvalue (Superstaq backend):", e_super)
    except Exception as exc:
        print("Superstaq VQE not run:", exc)

if __name__ == "__main__":
    main()
