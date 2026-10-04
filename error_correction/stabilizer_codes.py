import cirq
from cirq import Circuit, Gate, Qid, PauliString, Operation

from sympy import Symbol

from .protocols import QecProtocol


def _verify_clifford_circuit_is_identity(ct: Circuit):
    pass
    # TODO: implement efficiently.


class StabilizerSet:
    def __init__(self, stabilizers: list[str]):
        self.n_st = len(stabilizers)
        self.n = len(stabilizers[0])
        for st in stabilizers:
            assert len(st) == self.n
            for i in range(self.n):
                assert st[i] in ["I", "X", "Z"]

        # Fake qubits on which we'll apply stabilizers before we know real qubits.
        self.model_qubits = cirq.LineRange(self.n)

        # Convert stabilizers to Pauli string.
        self.stabilizers: list[PauliString] = []
        for st in stabilizers:
            ps: list[cirq.Operation] = []
            for i in range(self.n):
                if stabilizers[i] == "X":
                    ps.append(cirq.X(self.model_qubits[i]))
                elif stabilizers[i] == "Z":
                    ps.append(cirq.Z(self.model_qubits[i]))
            self.stabilizers.append(PauliString(ps))

        self.encoding_circuit: list[cirq.Operation] = []
        # TODO: pre-compute the circuit that encodes the state.
        # First we need to represent all n-k stabilizers as 2n-bit vectors,
        # put them in matrix and append all-X and all-Z operators.
        # Then, perform the symplectic Gaussian elimination to decompose this
        # into sequence of H, S and CNOT

        # Validate the encoding circuit.
        # Must be U Z_i U* = S_i
        # Equivalent to S_i U* Z_i U = I
        for i in range(self.n_st):
            ct = [self.stabilizers[i]]
            self.apply_decoding_circuit(ct, self.model_qubits)
            ct += cirq.Z()
            self.apply_encoding_circuit(ct, self.model_qubits)
            _verify_clifford_circuit_is_identity(ct)

    def get_stabilizers(self, qubits: list[Qid]):
        assert len(qubits) == self.n
        qubit_map = {self.model_qubits[i]: qubits[i] for i in range(self.n)}
        return [st.transform_qubits(qubit_map) for st in self.stabilizers]

    def apply_encoding_circuit(self, ct: Circuit, qubits: list[Qid]):
        """Applies U (decomposed into H/S/CNOT)."""
        assert len(qubits) == self.n
        qubit_map = {self.model_qubits[i]: qubits[i] for i in range(self.n)}
        for op in self.encoding_circuit:
            ct += op.transform_qubits(qubit_map)

    def apply_decoding_circuit(self, ct: Circuit, qubits: list[Qid]):
        """Applies U* (decomposed into H/S*/CNOT)."""
        assert len(qubits) == self.n
        qubit_map = {self.model_qubits[i]: qubits[i] for i in range(self.n)}
        for op in self.encoding_circuit[::-1]:
            ct += (op**-1).transform_qubits(qubit_map)


class StabilizerCode(QecProtocol):
    def __init__(
        self,
        physical_qubits: int,
        logical_qubits: int,
        distance: int,
        stabilizers: list[str],
        name: str | None = None,
    ):
        self.n = physical_qubits
        self.k = logical_qubits
        self.d = distance
        self.n_st = self.n - self.k  # Number of stablizers

        self.stabilizers = StabilizerSet(stabilizers)
        if self.stabilizers != self.n_st:
            raise ValueError("Wrong number of stabilizers")
        if self.stabilizers.n != self.n:
            raise ValueError(f"Wrong length of the stabilizer, must be {self.n}")

        self.meas_key_counter = 0

        Operation
        self.encoding_circuit: list[tuple[Gate, int]] = []

    def _prepare_encoding_circuit(self):
        pass

    def signature(self):
        return f"[[{self.n},{self.k},{self.d}]]"

    def _allocate_qubits(self, circuit, num_qubits):
        n0 = len(circuit.all_qubits())
        return [cirq.NamedQubit(f"aux_{n0+i}") for i in range(num_qubits)]

    def encode(self, ct: Circuit, qubits: list[Qid]) -> list[Qid]:
        assert len(qubits) == self.k

        # Create n-k additional qubits.
        aux = self._allocate_qubits(ct, self.n_st)

        # Apply the encoding circuit.
        self.stabilizers.apply_encoding_circuit(ct, qubits + aux)

        return qubits + aux

    def _measure_stabilizer(
        self, ct: Circuit, stabilizer: PauliString, anc: Qid
    ) -> Symbol:
        """Adds a circuit to measure stabilizer using given ancilla.

        Returns measurment result as symbol.
        Resets the ancilla.
        """

        # TODO: apply all controlled gates from qubits to anc so that after
        # measurment anc in computational basis we get stabilizer measurment.

        key = f"m{self.meas_key_counter}"
        self.meas_key_counter += 1
        ct += cirq.measure(anc, key=key)
        ct += cirq.reset(anc)
        return Symbol(key)

    def decode(self, ct: Circuit, qubits: list[Qid]) -> list[Qid]:
        assert len(qubits) == self.n

        anc = self._allocate_qubits(ct, 1)[0]

        # Syndrome measurments.
        syndrome = []
        for i in range(self.n):
            st = self.stabilizers.get_stabilizer_on_qubits(i, qubits)
            result = self._measure_stabilizer(ct, st, anc)
            syndrome.append(result)

        # TODO: add code to apply error correction based on symbolic syndromes.
        # This will use some controlled X and Z gates.

        # Decode.
        self.stabilizers.apply_decoding_circuit(ct, qubits)

        return qubits[0 : self.k]


# https://errorcorrectionzoo.org/c/stab_5_1_3
FIVE_QUBIT_PERFECT_CODE = StabilizerCode(
    5,
    1,
    3,
    ["XZZXI", "IXZZX", "XIXZZ", "ZXIXZ"],
)
