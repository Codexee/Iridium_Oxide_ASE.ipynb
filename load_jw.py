import os
import json
import warnings
import numpy as np
from pathlib import Path
from typing import Iterable, Union, Any, Optional
import traceback

from qiskit.quantum_info import SparsePauliOp
from qiskit.circuit.library import TwoLocal
from qiskit_algorithms.minimum_eigensolvers import VQE
from qiskit_algorithms.optimizers import SLSQP
from qiskit_algorithms.exceptions import AlgorithmError
from qiskit_aer.primitives import Estimator as AerEstimator


# --------------------------
# Load JW JSON
# --------------------------
def load_jw_json(json_path: Union[str, Path, Iterable[Union[str, Path]]]):
    if isinstance(json_path, (str, Path)):
        candidates = [json_path]
    else:
        candidates = list(json_path)

    last_err = None
    tried = []

    for p in candidates:
        p = Path(p)
        tried.append(str(p))
        try:
            with p.open("r") as f:
                jw = json.load(f)
            return jw, str(p)
        except OSError as e:
            last_err = e

    raise FileNotFoundError(
        f"Could not open JW JSON. Tried:\n"
        + "\n".join(f"  - {t}" for t in tried)
        + (f"\nLast error: {last_err}" if last_err else "")
    )


# --------------------------
# JW JSON -> SparsePauliOp
# --------------------------
def jw_to_sparsepauliop(jw_obj: Any) -> SparsePauliOp:
    """
    Project format:
      {"n_qubits": N, "terms":[{"paulis":[[q,"X"],...], "coeff_real":..., "coeff_imag":...}, ...]}
    where paulis == [] means identity term.
    """
    if not (isinstance(jw_obj, dict) and "terms" in jw_obj and "n_qubits" in jw_obj):
        raise ValueError("Unsupported JW JSON schema: expected keys 'n_qubits' and 'terms'.")

    n = int(jw_obj["n_qubits"])

    def label_from_paulis(paulis_list: list) -> str:
        label = ["I"] * n
        for item in paulis_list:
            if not (isinstance(item, (list, tuple)) and len(item) == 2):
                raise ValueError(f"Bad pauli entry {item!r}; expected [qubit_index, 'X/Y/Z']")
            q, p = int(item[0]), str(item[1]).upper()
            if p not in ("I", "X", "Y", "Z"):
                raise ValueError(f"Invalid Pauli '{p}' in entry {item!r}")
            if q < 0 or q >= n:
                raise ValueError(f"Qubit index {q} out of range for n_qubits={n}")
            # Qiskit ordering: rightmost char is qubit 0
            label[n - 1 - q] = p
        return "".join(label)

    pairs: list[tuple[str, complex]] = []
    for term in jw_obj["terms"]:
        paulis_list = term.get("paulis", [])
        label = label_from_paulis(paulis_list)

        c = complex(float(term.get("coeff_real", 0.0)), float(term.get("coeff_imag", 0.0)))
        pairs.append((label, c))

    H = SparsePauliOp.from_list(pairs).simplify(atol=0)

    # Ensure coefficients are real up to tiny numerical noise
    H = SparsePauliOp(H.paulis, np.real_if_close(H.coeffs, tol=1e-12))
    if np.max(np.abs(np.imag(H.coeffs))) > 0:
        raise ValueError("Hamiltonian has genuinely complex coefficients; VQE energy would be complex.")

    return H.simplify(atol=0)


# --------------------------
# VQE
# --------------------------
def build_ansatz(num_qubits: int) -> TwoLocal:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=DeprecationWarning)
        return TwoLocal(
            num_qubits,
            rotation_blocks=["ry", "rz"],
            entanglement_blocks="cx",
            entanglement="linear",
            reps=1,   # CI-stable
        )


def run_vqe_local_aer(H: SparsePauliOp) -> float:
    estimator = AerEstimator()          # V1 estimator (stable with qiskit_algorithms.VQE)
    ansatz = build_ansatz(H.num_qubits)
    optimizer = SLSQP(maxiter=60)

    vqe = VQE(estimator=estimator, ansatz=ansatz, optimizer=optimizer)

    try:
        res = vqe.compute_minimum_eigenvalue(H)
    except AlgorithmError as e:
        print("\n=== VQE failed: root exception below ===")
        cause = e.__cause__
        if cause:
            traceback.print_exception(type(cause), cause, cause.__traceback__)
        else:
            print("No __cause__ attached. Full error:", repr(e))
        raise

    return float(np.real(res.eigenvalue))


def run_vqe_superstaq(H: SparsePauliOp, target: str = "ibmq_fez") -> Optional[float]:
    """
    Optional integration path (only if SUPERSTAQ_API_KEY is set).
    Keep this out of normal CI unless you explicitly want network tests.
    """
    api_key = os.getenv("SUPERSTAQ_API_KEY", "").strip()
    if not api_key:
        print("No SUPERSTAQ_API_KEY set; skipping Superstaq VQE.")
        return None

    # TODO: implement properly against qiskit-superstaq primitives/backends you intend to use.
    # For now, skip to avoid flaky CI.
    print("SUPERSTAQ_API_KEY present, but Superstaq VQE not implemented in CI-safe mode.")
    return None


# --------------------------
# main
# --------------------------
def main():
    candidates = [
        os.getenv("JW_JSON_PATH", "").strip(),
        "inputs/qubit_hamiltonian_jw.json",
        "qubit_hamiltonian_jw.json",
    ]
    candidates = [c for c in candidates if c]

    jw, path_used = load_jw_json(candidates)
    print(f"Loaded JW Hamiltonian from: {path_used}")

    H = jw_to_sparsepauliop(jw)
    print(f"n_qubits = {H.num_qubits}, n_terms = {len(H.paulis)}")

    e_local = run_vqe_local_aer(H)
    print("VQE minimum eigenvalue (local Aer):", e_local)

    e_super = run_vqe_superstaq(H, target="ibmq_fez")
    if e_super is not None:
        print("VQE minimum eigenvalue (Superstaq):", e_super)


if __name__ == "__main__":
    main()
