import itertools

import cirq
import numpy as np
import sympy
from cirq import Circuit, PauliString, Qid

from .protocols import QecProtocol


def _verify_clifford_circuit_is_identity(ct: Circuit):
    operations = list(ct.all_operations())
    if not operations:
        return

    qubits = sorted(ct.all_qubits())
    tableau = cirq.CliffordGate.from_op_list(operations, qubits).clifford_tableau
    identity = cirq.CliffordTableau(len(qubits))
    if not (
        np.array_equal(tableau.xs, identity.xs)
        and np.array_equal(tableau.zs, identity.zs)
        and np.array_equal(tableau.rs, identity.rs)
    ):
        raise ValueError("Circuit is not the identity Clifford operation")


def _gf2_rref(matrix: np.ndarray) -> tuple[np.ndarray, list[int]]:
    matrix = matrix.copy().astype(np.uint8)
    pivots = []
    row = 0
    for column in range(matrix.shape[1]):
        pivot_rows = np.flatnonzero(matrix[row:, column])
        if not len(pivot_rows):
            continue
        pivot = row + int(pivot_rows[0])
        matrix[[row, pivot]] = matrix[[pivot, row]]
        for other in range(matrix.shape[0]):
            if other != row and matrix[other, column]:
                matrix[other] ^= matrix[row]
        pivots.append(column)
        row += 1
        if row == matrix.shape[0]:
            break
    return matrix, pivots


def _gf2_solve(matrix: np.ndarray, result: np.ndarray) -> np.ndarray:
    augmented = np.column_stack((matrix, result))
    reduced, pivots = _gf2_rref(augmented)
    coefficient_columns = matrix.shape[1]
    if any(
        not reduced[row, :coefficient_columns].any()
        and reduced[row, coefficient_columns]
        for row in range(reduced.shape[0])
    ):
        raise ValueError("Binary linear system has no solution")

    solution = np.zeros(coefficient_columns, dtype=np.uint8)
    for row, pivot in enumerate(pivots):
        if pivot < coefficient_columns:
            solution[pivot] = reduced[row, coefficient_columns]
    return solution


def _gf2_null_space(matrix: np.ndarray) -> list[np.ndarray]:
    reduced, pivots = _gf2_rref(matrix)
    free_columns = [i for i in range(matrix.shape[1]) if i not in pivots]
    basis = []
    for free in free_columns:
        vector = np.zeros(matrix.shape[1], dtype=np.uint8)
        vector[free] = 1
        for row, pivot in enumerate(pivots):
            vector[pivot] = reduced[row, free]
        basis.append(vector)
    return basis


def _symplectic_product(left: np.ndarray, right: np.ndarray) -> int:
    n = len(left) // 2
    return int((left[:n] @ right[n:] + left[n:] @ right[:n]) % 2)


class StabilizerSet:
    def __init__(self, stabilizers: list[str]):
        if not stabilizers:
            raise ValueError("At least one stabilizer is required")

        self.n_st = len(stabilizers)
        self.n = len(stabilizers[0])
        if self.n_st > self.n:
            raise ValueError("There cannot be more stabilizers than qubits")

        normalized = [st.upper() for st in stabilizers]
        for stabilizer in normalized:
            if len(stabilizer) != self.n:
                raise ValueError("All stabilizers must have the same length")
            if any(pauli not in "IXYZ" for pauli in stabilizer):
                raise ValueError("Stabilizers may contain only I, X, Y, and Z")

        self.model_qubits = list(cirq.LineQubit.range(self.n))
        self.binary_stabilizers = np.array(
            [self._to_binary(stabilizer) for stabilizer in normalized],
            dtype=np.uint8,
        )
        self._validate_generators(normalized)

        self.stabilizers = [
            PauliString(
                {
                    self.model_qubits[i]: getattr(cirq, pauli)
                    for i, pauli in enumerate(stabilizer)
                    if pauli != "I"
                }
            )
            for stabilizer in normalized
        ]
        self.encoding_circuit = self._prepare_encoding_circuit()
        self._validate_encoding_circuit()

    def _to_binary(self, stabilizer: str) -> np.ndarray:
        vector = np.zeros(2 * self.n, dtype=np.uint8)
        for i, pauli in enumerate(stabilizer):
            vector[i] = pauli in "XY"
            vector[self.n + i] = pauli in "ZY"
        return vector

    def _validate_generators(self, stabilizers: list[str]) -> None:
        _, pivots = _gf2_rref(self.binary_stabilizers)
        if len(pivots) != self.n_st:
            raise ValueError("Stabilizer generators must be independent")
        for i in range(self.n_st):
            for j in range(i):
                if _symplectic_product(
                    self.binary_stabilizers[i], self.binary_stabilizers[j]
                ):
                    raise ValueError(
                        f"Stabilizer generators {stabilizers[j]!r} and "
                        f"{stabilizers[i]!r} must commute"
                    )

    def _prepare_encoding_circuit(self) -> list[cirq.Operation]:
        stabilizers = self.binary_stabilizers
        dual_constraints = np.hstack(
            (stabilizers[:, self.n :], stabilizers[:, : self.n])
        )
        duals = [
            _gf2_solve(
                dual_constraints,
                np.eye(self.n_st, dtype=np.uint8)[i],
            )
            for i in range(self.n_st)
        ]

        for i in range(self.n_st):
            for j in range(i):
                if _symplectic_product(duals[i], duals[j]):
                    duals[i] ^= stabilizers[j]

        complement_constraints = []
        for vector in itertools.chain(stabilizers, duals):
            complement_constraints.append(
                np.concatenate((vector[self.n :], vector[: self.n]))
            )
        complement = _gf2_null_space(np.array(complement_constraints))

        logical_xs = []
        logical_zs = []
        while complement:
            logical_x = complement.pop(0)
            partner = next(
                (
                    i
                    for i, vector in enumerate(complement)
                    if _symplectic_product(logical_x, vector)
                ),
                None,
            )
            if partner is None:
                raise ValueError("Could not complete the stabilizer symplectic basis")
            logical_z = complement.pop(partner)
            logical_xs.append(logical_x)
            logical_zs.append(logical_z)
            complement = [
                vector
                ^ (_symplectic_product(vector, logical_z) * logical_x)
                ^ (_symplectic_product(vector, logical_x) * logical_z)
                for vector in complement
            ]

        k = self.n - self.n_st
        if len(logical_xs) != k:
            raise ValueError("Stabilizers do not define the expected code space")

        x_images = logical_xs + duals
        z_images = logical_zs + list(stabilizers)
        images = np.array(x_images + z_images, dtype=np.uint8)
        tableau = cirq.CliffordTableau(
            self.n,
            rs=np.zeros(2 * self.n, dtype=bool),
            xs=images[:, : self.n].astype(bool),
            zs=images[:, self.n :].astype(bool),
        )
        return cirq.decompose_clifford_tableau_to_operations(self.model_qubits, tableau)

    def _validate_encoding_circuit(self):
        k = self.n - self.n_st
        for i, stabilizer in enumerate(self.stabilizers):
            circuit = Circuit(stabilizer)
            self.apply_decoding_circuit(circuit, self.model_qubits)
            circuit.append(cirq.Z(self.model_qubits[k + i]))
            self.apply_encoding_circuit(circuit, self.model_qubits)
            _verify_clifford_circuit_is_identity(circuit)

    def get_stabilizers(self, qubits: list[Qid]) -> list[PauliString]:
        if len(qubits) != self.n:
            raise ValueError(f"Expected {self.n} qubits, got {len(qubits)}")
        qubit_map = dict(zip(self.model_qubits, qubits))
        return [st.transform_qubits(qubit_map) for st in self.stabilizers]

    def apply_encoding_circuit(self, ct: Circuit, qubits: list[Qid]):
        """Applies the encoding Clifford U."""
        if len(qubits) != self.n:
            raise ValueError(f"Expected {self.n} qubits, got {len(qubits)}")
        qubit_map = dict(zip(self.model_qubits, qubits))
        ct.append(op.transform_qubits(qubit_map) for op in self.encoding_circuit)

    def apply_decoding_circuit(self, ct: Circuit, qubits: list[Qid]):
        """Applies the inverse encoding Clifford U*."""
        if len(qubits) != self.n:
            raise ValueError(f"Expected {self.n} qubits, got {len(qubits)}")
        qubit_map = dict(zip(self.model_qubits, qubits))
        ct.append(
            (op**-1).transform_qubits(qubit_map)
            for op in reversed(self.encoding_circuit)
        )


class StabilizerCode(QecProtocol):
    def __init__(
        self,
        physical_qubits: int,
        logical_qubits: int,
        distance: int,
        stabilizers: list[str],
        name: str | None = None,
    ):
        if not 0 < logical_qubits < physical_qubits:
            raise ValueError("Expected 0 < logical_qubits < physical_qubits")
        if distance < 1:
            raise ValueError("Distance must be positive")

        self.n = physical_qubits
        self.k = logical_qubits
        self.d = distance
        self.n_st = self.n - self.k
        self.name = name or f"{self.signature()} stabilizer code"

        self.stabilizers = StabilizerSet(stabilizers)
        if self.stabilizers.n_st != self.n_st:
            raise ValueError(f"Expected {self.n_st} stabilizers")
        if self.stabilizers.n != self.n:
            raise ValueError(f"Stabilizers must act on {self.n} qubits")

        self.meas_key_counter = 0
        self.encoding_circuit = self.stabilizers.encoding_circuit
        self.syndrome_corrections = self._prepare_syndrome_corrections()

    def _prepare_encoding_circuit(self):
        self.encoding_circuit = self.stabilizers._prepare_encoding_circuit()

    def _prepare_syndrome_corrections(self) -> dict[tuple[int, ...], str]:
        corrections = {}
        max_weight = (self.d - 1) // 2
        for weight in range(1, max_weight + 1):
            for locations in itertools.combinations(range(self.n), weight):
                for paulis in itertools.product("XYZ", repeat=weight):
                    error = ["I"] * self.n
                    for location, pauli in zip(locations, paulis):
                        error[location] = pauli
                    error_string = "".join(error)
                    error_vector = self.stabilizers._to_binary(error_string)
                    syndrome = tuple(
                        _symplectic_product(error_vector, stabilizer)
                        for stabilizer in self.stabilizers.binary_stabilizers
                    )
                    if any(syndrome):
                        corrections.setdefault(syndrome, error_string)
        return corrections

    def signature(self):
        return f"[[{self.n},{self.k},{self.d}]]"

    def _allocate_qubits(self, circuit: Circuit, num_qubits: int) -> list[Qid]:
        existing = circuit.all_qubits()
        result = []
        index = len(existing)
        while len(result) < num_qubits:
            qubit = cirq.NamedQubit(f"aux_{index}")
            if qubit not in existing:
                result.append(qubit)
            index += 1
        return result

    def encode(self, ct: Circuit, qubits: list[Qid]) -> list[Qid]:
        if len(qubits) != self.k:
            raise ValueError(f"Expected {self.k} logical qubits, got {len(qubits)}")

        aux = self._allocate_qubits(ct, self.n_st)
        physical_qubits = qubits + aux
        self.stabilizers.apply_encoding_circuit(ct, physical_qubits)
        return physical_qubits

    def _measure_stabilizer(
        self, ct: Circuit, stabilizer: PauliString, anc: Qid
    ) -> sympy.Symbol:
        """Measure a Pauli stabilizer with an ancilla and reset the ancilla."""
        ct.append(cirq.H(anc))
        for qubit, pauli in stabilizer.items():
            ct.append(pauli.on(qubit).controlled_by(anc))
        ct.append(cirq.H(anc))

        key = f"m{self.meas_key_counter}"
        self.meas_key_counter += 1
        ct.append(cirq.measure(anc, key=key))
        ct.append(cirq.reset(anc))
        return sympy.Symbol(key)

    def decode(self, ct: Circuit, qubits: list[Qid]) -> list[Qid]:
        if len(qubits) != self.n:
            raise ValueError(f"Expected {self.n} physical qubits, got {len(qubits)}")

        anc = self._allocate_qubits(ct, 1)[0]
        syndrome_symbols = [
            self._measure_stabilizer(ct, stabilizer, anc)
            for stabilizer in self.stabilizers.get_stabilizers(qubits)
        ]

        for syndrome, correction in self.syndrome_corrections.items():
            condition = sympy.And(
                *(
                    symbol if bit else sympy.Not(symbol)
                    for symbol, bit in zip(syndrome_symbols, syndrome)
                )
            )
            for qubit, pauli in zip(qubits, correction):
                if pauli != "I":
                    ct.append(
                        getattr(cirq, pauli)(qubit).with_classical_controls(condition)
                    )

        self.stabilizers.apply_decoding_circuit(ct, qubits)
        return qubits[: self.k]


# https://errorcorrectionzoo.org/c/stab_5_1_3
FIVE_QUBIT_PERFECT_CODE = StabilizerCode(
    5,
    1,
    3,
    ["XZZXI", "IXZZX", "XIXZZ", "ZXIXZ"],
)

SHOR_CODE = StabilizerCode(
    9,
    1,
    3,
    [
        "ZZIIIIIII",
        "IZZIIIIII",
        "IIIZZIIII",
        "IIIIZZIII",
        "IIIIIIZZI",
        "IIIIIIIZZ",
        "XXXXXXIII",
        "IIIXXXXXX",
    ],
)
