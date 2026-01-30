import os
import json
import warnings
import numpy as np
from pathlib import Path
from typing import Iterable, Union, Any

from qiskit.quantum_info import SparsePauliOp
from qiskit.circuit.library import TwoLocal
from qiskit_algorithms.minimum_eigensolvers import VQE
from qiskit_algorithms.optimizers import COBYLA
from __future__ import annotations

# load JW JSON

def load_jw_json(json_path: Union[str, Path, Iterable[Union[str, Path]]]):
    # Allow passing a single path or a list/tuple of paths
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
        f"Could not open {Path(tried[0]).name if tried else 'JW JSON'}. Tried:\n"
        + "\n".join(f"  - {t}" for t in tried)
        + (f"\nLast error: {last_err}" if last_err else "")
    )

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

    api_key = os.getenv("SUPERSTAQ_API_KEY", "").strip()
    if not api_key:
        print("No SUPERSTAQ_API_KEY set; skipping Superstaq VQE.")
        return None
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

def _to_complex(x):
    """Convert common JSON coefficient encodings to complex."""
    if isinstance(x, (int, float, complex, np.number)):
        return complex(x)
    if isinstance(x, (list, tuple)) and len(x) == 2:
        return complex(float(x[0]), float(x[1]))
    if isinstance(x, dict):
        # common patterns: {"real":..., "imag":...} or {"re":..., "im":...}
        if "real" in x or "imag" in x:
            return complex(float(x.get("real", 0.0)), float(x.get("imag", 0.0)))
        if "re" in x or "im" in x:
            return complex(float(x.get("re", 0.0)), float(x.get("im", 0.0)))
        # sometimes {"value":[re,im]}
        if "value" in x:
            return _to_complex(x["value"])
    raise TypeError(f"Unrecognized coefficient encoding: {x!r}")

def jw_to_sparsepauliop(jw_obj: Any) -> SparsePauliOp:
    """
    Convert JW Hamiltonian JSON into Qiskit's SparsePauliOp.

    Supports:
      A) The project format:
         {
           "n_qubits": 14,
           "terms": [{"paulis": [[0,"Z"], [3,"X"], ...], "coeff_real": ..., "coeff_imag": ...}, ...]
         }
         NOTE: "paulis": [] means identity term.
      B) Label string formats:
         {"paulis": ["IZX", ...], "coeffs": [...]}
         {"terms": [{"pauli": "IZX", "coeff": ...}, ...]}
         [["IZX", coeff], ...]
    """
    # ---------- helpers ----------
    def _to_complex(x: Any) -> complex:
        if isinstance(x, (int, float, complex, np.number)):
            return complex(x)
        if isinstance(x, (list, tuple)) and len(x) == 2:
            return complex(float(x[0]), float(x[1]))
        if isinstance(x, dict):
            if "real" in x or "imag" in x:
                return complex(float(x.get("real", 0.0)), float(x.get("imag", 0.0)))
            if "re" in x or "im" in x:
                return complex(float(x.get("re", 0.0)), float(x.get("im", 0.0)))
            if "value" in x:
                return _to_complex(x["value"])
        raise TypeError(f"Unrecognized coefficient encoding: {x!r}")

    def _label_from_sparse_paulis(paulis: list, n_qubits: int) -> str:
        # Qiskit Pauli strings are ordered with qubit (n-1) on the left and qubit 0 on the right.
        label = ["I"] * n_qubits
        for item in paulis:
            if not (isinstance(item, (list, tuple)) and len(item) == 2):
                raise ValueError(f"Bad pauli entry {item!r}; expected [qubit_index, 'X/Y/Z']")
            q, p = item
            q = int(q)
            p = str(p).upper()
            if q < 0 or q >= n_qubits:
                raise ValueError(f"Qubit index {q} out of range for n_qubits={n_qubits}")
            if p not in ("I", "X", "Y", "Z"):
                raise ValueError(f"Invalid Pauli '{p}' in entry {item!r}")
            label[n_qubits - 1 - q] = p
        return "".join(label)

    # ---------- detect your project format first ----------
    if isinstance(jw_obj, dict) and "terms" in jw_obj and "n_qubits" in jw_obj:
        n = int(jw_obj["n_qubits"])
        labels: list[str] = []
        coeffs: list[complex] = []

        for term in jw_obj["terms"]:
            if not isinstance(term, dict):
                raise TypeError(f"Unexpected term type {type(term)}; expected dict.")
            paulis = term.get("paulis", [])
            # paulis may be [] -> identity term (valid)
            label = _label_from_sparse_paulis(paulis, n)

            # coefficients in your JSON are split real/imag
            if "coeff_real" in term or "coeff_imag" in term:
                c = complex(float(term.get("coeff_real", 0.0)), float(term.get("coeff_imag", 0.0)))
            else:
                # fallback if someone stores "coeff" differently
                c = _to_complex(term.get("coeff", term.get("coefficient", term.get("c"))))

            labels.append(label)
            coeffs.append(c)

        H = SparsePauliOp.from_list(list(zip(labels, coeffs)))
        return H.simplify(atol=0)  # combine duplicates safely

    # ---------- otherwise: label-string formats ----------
    terms = None
    if isinstance(jw_obj, dict):
        if "paulis" in jw_obj and "coeffs" in jw_obj:
            terms = list(zip(jw_obj["paulis"], jw_obj["coeffs"]))
        elif "terms" in jw_obj:
            terms = jw_obj["terms"]
        elif "operators" in jw_obj:
            terms = jw_obj["operators"]
        else:
            raise ValueError(f"Unrecognized JW dict keys: {list(jw_obj.keys())[:20]}")
    elif isinstance(jw_obj, list):
        terms = jw_obj
    else:
        raise TypeError(f"Unsupported JW JSON type: {type(jw_obj)}")

    pairs: list[tuple[str, complex]] = []
    max_len = 0

    for t in terms:
        if isinstance(t, dict):
            label = t.get("pauli") or t.get("p") or t.get("label")
            coeff = t.get("coeff") if "coeff" in t else t.get("coefficient", t.get("c"))
        elif isinstance(t, (list, tuple)) and len(t) == 2:
            label, coeff = t
        else:
            raise TypeError(f"Unrecognized term entry: {t!r}")

        if label is None:
            raise ValueError(f"Missing pauli label in term: {t!r}")

        label = str(label).replace(" ", "").upper()
        max_len = max(max_len, len(label))
        pairs.append((label, _to_complex(coeff)))

    padded = [(("I" * (max_len - len(lbl)) + lbl), c) for (lbl, c) in pairs]
    return SparsePauliOp.from_list(padded).simplify(atol=0)

def main():
    candidates = [
        os.getenv("JW_JSON_PATH", "").strip(),
        "inputs/qubit_hamiltonian_jw.json",
        "qubit_hamiltonian_jw.json",
    ]
    candidates = [c for c in candidates if c]  # drop empty strings

    jw, path_used = load_jw_json(candidates)
    print(f"Loaded JW Hamiltonian from: {path_used}")
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
