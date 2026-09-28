"""Sparse, single-shot simulator for a restricted subset of Cirq circuits.

The simulator keeps the quantum state as a sparse list of computational basis
states and their amplitudes instead of a dense state vector. This is efficient
for circuits that mostly preserve sparsity (for example arithmetic/reversible
circuits with occasional single-qubit superposition gates).

Execution model:

1. Start in ``|0...0>`` represented as one basis state with amplitude ``1``.
2. Apply operations one by one, updating sparse basis states/amplitudes.
3. Measurements collapse the sparse state stochastically according to Born rule.
4. Return a single-shot ``cirq.ResultDict``.

This simulator is intentionally limited and aimed at tests and debugging.
Unsupported operations raise ``ValueError``.
"""

import math
import random
from collections.abc import Callable, Iterable, Sequence

import cirq
import numpy as np
import sympy

_AMPLITUDE_EPS = 1e-9


class SparseSimArithmeticGate(cirq.Gate):
    """Test-only gate that represents a reversible arithmetic basis-state map.

    States are interpreted as one or more little-endian integer registers.
    ``f`` maps those integer register values to new register values.
    ``f`` must represent a reversible transformation, but reversibility is not
    checked by this class.

    Can only be simulated with SparseSimulator.
    """

    def __init__(
        self,
        input_sizes: list[int],
        f: Callable[..., tuple[int, ...]],
    ) -> None:
        """Initialize the gate."""
        self.input_sizes = input_sizes
        self.f = f

    def num_qubits(self) -> int:
        """The number of qubits this gate acts on."""
        return sum(self.input_sizes)


class SparseSimQubitManager(cirq.QubitManager):
    """Simple qubit manager with allocation/release over ``LineQubit`` indices."""

    def __init__(self, simulator: "SparseSimulator") -> None:
        """Initialize the simulator."""
        self.simulator = simulator
        self.num_qubits = 0
        self.free_qubits: list[int] = []
        self.free_qubits_set: set[int] = set()

    def _allocate_qubit(self) -> cirq.LineQubit:
        if len(self.free_qubits) == 0:
            qubit_id = self.num_qubits
            self.num_qubits += 1
        else:
            qubit_id = self.free_qubits.pop()
            self.free_qubits_set.remove(qubit_id)
        return cirq.LineQubit(qubit_id)

    def qalloc(self, n: int, dim: int = 2) -> list[cirq.Qid]:
        """Allocate qubits."""
        assert dim == 2
        return [self._allocate_qubit() for _ in range(n)]

    def qborrow(self, n: int, dim: int = 2) -> list[cirq.Qid]:
        """Not implemented."""
        raise NotImplementedError("qborrow is not supported")

    def qfree(self, qubits: Iterable[cirq.Qid]) -> None:
        """Free qubits."""
        for q in qubits:
            assert isinstance(q, cirq.LineQubit)
            assert 0 <= q.x < self.num_qubits
            if q.x in self.free_qubits_set:
                raise RuntimeError(f"Qubit {q.x} released twice")
            if not self.simulator._is_qubit_zero(q.x):
                raise RuntimeError(f"Qubit {q.x} released not in zero state")
            self.free_qubits.append(q.x)
            self.free_qubits_set.add(q.x)

    def num_allocated_qubits(self) -> int:
        """Returns number of allocated qubits."""
        return self.num_qubits - len(self.free_qubits)


def _apply_cx(state: int, control_bit: int, target_bit: int) -> int:
    if (state >> control_bit) % 2 == 1:
        return state ^ (1 << target_bit)
    return state


def _apply_ccx(state: int, ctrl1: int, ctrl2: int, target: int) -> int:
    if (state >> ctrl1) % 2 == 1 and (state >> ctrl2) % 2 == 1:
        return state ^ (1 << target)
    return state


def _cz_phase(state: int, ctrl: int, target: int, exponent: float) -> complex:
    if (state >> ctrl) % 2 == 1 and (state >> target) % 2 == 1:
        return complex(math.cos(math.pi * exponent), math.sin(math.pi * exponent))
    return 1.0 + 0.0j


def _normalize(amplitudes: list[complex]) -> list[complex]:
    norm = math.sqrt(sum(abs(a) ** 2 for a in amplitudes))
    return [a / norm for a in amplitudes]


class SparseSimulator:
    """Sparse simulator for a restricted set of Cirq operations.

    Supported operations include:
      - ``X``, ``CNOT``, ``CCNOT`` (basis permutation).
      - ``CZPowGate`` with numeric exponent (phase on ``|11>`` branch).
      - ``SparseSimArithmeticGate``.
      - ``MeasurementGate`` and ``ResetChannel``.
      - ``ClassicallyControlledOperation`` with ``KeyCondition`` or ``SympyCondition``.
      - Any single-qubit gate with known unitary.

    Other operations are attempted via one-step decomposition and processed
    recursively. If an operation still cannot be handled, simulation fails.

    Usage:
      - Create a simulator instance: ``sim = SparseSimulator()``.
      - Allocate qubits using ``sim.qubit_manager.qalloc``. Qubits created in any other
        way are not supported.
      - Create a circuit.
      - Call ``sim.run(circuit)``.
      - Use ``sim.read_register`` to read the value stored in a given register,
        interpreted as an unsigned little-endian integer.
      - Inspect measurement results in the ``cirq.Result`` object returned by
        ``sim.run``.

    Notes:
      - ``run`` is single-shot only.
      - ``measurement_results`` stores the latest integer value per measurement key.
        These values are little-endian; classical conditions use Cirq's big-endian
        integers and measurement occurrence indices.
      - ``read_register`` requires the final state to be a single basis state.
        If it's not the case, add measurements.

    """

    def __init__(self, seed: int | None = None) -> None:
        """Initialize SparseSimulator."""
        self.random = random.Random(seed)
        self.qubit_manager = SparseSimQubitManager(self)

        # Sparse state is represented by a list of basis states with
        # non-zero amplitudes and a list of amplitudes.
        self.basis_states = [0]
        self.amplitudes = [1.0 + 0.0j]

        # Measurement results.
        self.measurement_results: dict[str, int] = {}
        self._measurement_history: dict[str, list[int]] = {}

        # Length of measurements (to restore measurements as bit vectors).
        self._meas_len: dict[str, int] = {}

    def _apply_x(self, qid: int) -> None:
        self.basis_states = [s ^ (1 << qid) for s in self.basis_states]

    def _apply_cx(self, ctrl: int, target: int) -> None:
        self.basis_states = [_apply_cx(s, ctrl, target) for s in self.basis_states]

    def _apply_ccx(self, ctrl1: int, ctrl2: int, target: int) -> None:
        self.basis_states = [
            _apply_ccx(s, ctrl1, ctrl2, target) for s in self.basis_states
        ]

    def _apply_cz(self, ctrl: int, target: int, exponent: float) -> None:
        self.amplitudes = [
            amplitude * _cz_phase(state, ctrl, target, exponent)
            for state, amplitude in zip(self.basis_states, self.amplitudes, strict=True)
        ]

    def _apply_global_shift(self, gate: cirq.EigenGate) -> None:
        if gate.global_shift == 0:
            return
        angle = math.pi * float(gate.exponent) * gate.global_shift
        phase = complex(math.cos(angle), math.sin(angle))
        self.amplitudes = [amplitude * phase for amplitude in self.amplitudes]

    def _apply_arithmetic_gate(
        self, op_gate: SparseSimArithmeticGate, qubits: list[int]
    ) -> None:
        assert len(qubits) == op_gate.num_qubits()
        registers: list[list[int]] = []
        offset = 0
        for n in op_gate.input_sizes:
            registers.append(qubits[offset : offset + n])
            offset += n
        new_basis_states = []
        for s in self.basis_states:
            inputs = [
                sum(((s >> q) & 1) << i for i, q in enumerate(r)) for r in registers
            ]
            outputs = op_gate.f(*inputs)
            assert len(outputs) == len(inputs)
            new_state = s
            for inp, out, reg in zip(inputs, outputs, registers, strict=True):
                xor_val = inp ^ out
                for i, q in enumerate(reg):
                    if (xor_val >> i) & 1:
                        new_state ^= 1 << q
            new_basis_states.append(new_state)
        self.basis_states = new_basis_states

    def _apply_single_qubit_unitary_gate(
        self, op_gate: cirq.Gate, qubit_id: int
    ) -> None:
        """Apply a 2x2 unitary to one qubit across all sparse basis states.

        Output contributions that land on the same basis state are summed.
        Terms with very small amplitudes (``<_AMPLITUDE_EPS``) are dropped.
        """
        matrix = cirq.unitary(op_gate, default=None)
        assert matrix is not None
        assert matrix.shape == (2, 2)

        merged_amplitudes: dict[int, complex] = {}
        for state, amplitude in zip(self.basis_states, self.amplitudes, strict=True):
            input_bit = (state >> qubit_id) & 1
            for output_bit in (0, 1):
                coeff = matrix[output_bit, input_bit]
                if coeff == 0:
                    continue
                new_state = (
                    state if output_bit == input_bit else state ^ (1 << qubit_id)
                )
                merged_amplitudes[new_state] = (
                    merged_amplitudes.get(new_state, 0.0 + 0.0j) + amplitude * coeff
                )

        kept_states = [
            (state, amplitude)
            for state, amplitude in merged_amplitudes.items()
            if abs(amplitude) >= _AMPLITUDE_EPS
        ]
        assert len(kept_states) > 0
        self.basis_states = [state for state, _ in kept_states]
        self.amplitudes = _normalize([amplitude for _, amplitude in kept_states])

    def _measure_qubit(self, target_qubit: int) -> int:
        """Measure one qubit and collapse sparse state according to Born rule.

        Returns the measured bit (0/1). If all current basis states already
        agree on the measured bit, this is a cheap no-op collapse.
        """
        if len(self.basis_states) == 1:
            return (self.basis_states[0] >> target_qubit) & 1

        zero_subspace: list[tuple[int, complex]] = []
        one_subspace: list[tuple[int, complex]] = []
        for state, amplitude in zip(self.basis_states, self.amplitudes, strict=True):
            if (state >> target_qubit) & 1:
                one_subspace.append((state, amplitude))
            else:
                zero_subspace.append((state, amplitude))

        if len(one_subspace) == 0:
            return 0
        if len(zero_subspace) == 0:
            return 1

        zero_prob = sum(abs(amplitude) ** 2 for _, amplitude in zero_subspace)
        one_prob = sum(abs(amplitude) ** 2 for _, amplitude in one_subspace)
        assert zero_prob > 0
        assert one_prob > 0

        if self.random.random() < zero_prob / (zero_prob + one_prob):
            measured_bit = 0
            chosen = zero_subspace
        else:
            measured_bit = 1
            chosen = one_subspace

        self.basis_states = [state for state, _ in chosen]
        self.amplitudes = _normalize([amplitude for _, amplitude in chosen])
        return measured_bit

    def _apply_measurement_gate(
        self, op: cirq.Operation, op_gate: cirq.MeasurementGate, qubit_ids: list[int]
    ) -> None:
        """Apply a measurement gate and record key result as little-endian int."""
        key = cirq.measurement_key_name(op)
        measured_value = 0
        measured_bits = []
        for i, qid in enumerate(qubit_ids):
            measured_bit = self._measure_qubit(qid)
            invert = i < len(op_gate.invert_mask) and op_gate.invert_mask[i]
            if invert:
                measured_bit ^= 1
            measured_value |= measured_bit << i
            measured_bits.append(measured_bit)
        self.measurement_results[key] = measured_value
        self._meas_len[key] = len(qubit_ids)
        self._measurement_history.setdefault(key, []).append(
            cirq.big_endian_bits_to_int(measured_bits)
        )

    def _apply_reset(self, target_qubit: int) -> None:
        """Reset one qubit to ``|0>`` via measure-and-conditional-flip."""
        if self._measure_qubit(target_qubit) == 1:
            self._apply_x(target_qubit)

    def _resolve_sympy_condition(self, condition: cirq.SympyCondition) -> bool:
        replacements: dict[str, object] = {}
        for symbol in condition.expr.free_symbols:
            if isinstance(symbol, sympy.Symbol):
                replacements[symbol.name] = self._measurement_history[symbol.name][-1]
        value = condition.expr.subs(replacements)
        assert len(value.free_symbols) == 0
        return bool(value)

    def _classical_condition_satisfied(self, condition: cirq.Condition) -> bool:
        """Evaluate a supported classical condition using stored measurements."""
        if isinstance(condition, cirq.SympyCondition):
            return self._resolve_sympy_condition(condition)
        if isinstance(condition, cirq.KeyCondition):
            return self._measurement_history[str(condition.key)][condition.index] != 0
        raise ValueError(f"Unsupported classical condition: {condition}")

    def _run_op(self, op: cirq.Operation, context: cirq.DecompositionContext) -> None:
        """Execute one operation, recursively decomposing when needed."""
        if isinstance(op, cirq.ClassicallyControlledOperation):
            if all(
                self._classical_condition_satisfied(condition)
                for condition in op.classical_controls
            ):
                self._run_op(op.without_classical_controls(), context)
            return

        op_gate = op.gate
        qubit_ids = [q.x for q in op.qubits]

        if isinstance(op_gate, cirq.XPowGate) and op_gate.exponent == 1:
            self._apply_x(qubit_ids[0])
            self._apply_global_shift(op_gate)
            return
        if isinstance(op_gate, cirq.CXPowGate) and op_gate.exponent == 1:
            self._apply_cx(qubit_ids[0], qubit_ids[1])
            self._apply_global_shift(op_gate)
            return
        if isinstance(op_gate, cirq.CCXPowGate) and op_gate.exponent == 1:
            self._apply_ccx(qubit_ids[0], qubit_ids[1], qubit_ids[2])
            self._apply_global_shift(op_gate)
            return
        if isinstance(op_gate, cirq.CZPowGate) and not cirq.is_parameterized(op_gate):
            self._apply_cz(qubit_ids[0], qubit_ids[1], float(op_gate.exponent))
            self._apply_global_shift(op_gate)
            return
        if isinstance(op_gate, SparseSimArithmeticGate):
            self._apply_arithmetic_gate(op_gate, qubit_ids)
            return
        if isinstance(op_gate, cirq.MeasurementGate):
            self._apply_measurement_gate(op, op_gate, qubit_ids)
            return
        if isinstance(op_gate, cirq.ResetChannel):
            assert len(qubit_ids) == 1
            self._apply_reset(qubit_ids[0])
            return

        # Prefer explicit unitaries to decompositions introducing unsupported gates.
        if (
            op_gate is not None
            and len(qubit_ids) == 1
            and cirq.has_unitary(op_gate, allow_decompose=False)
        ):
            self._apply_single_qubit_unitary_gate(op_gate, qubit_ids[0])
            return

        num_alloc_before = self.qubit_manager.num_allocated_qubits()
        decomposed = cirq.decompose_once(
            op, default=None, context=context, flatten=False
        )
        if decomposed is not None:
            for sub_op in cirq.flatten_to_ops(decomposed):
                self._run_op(sub_op, context)
            if self.qubit_manager.num_allocated_qubits() != num_alloc_before:
                raise RuntimeError(f"Operation {op} did not free all allocated qubits")
            return

        raise ValueError(f"Operation cannot be simulated: {op}.")

    def run(self, circuit: cirq.Circuit) -> cirq.Result:
        """Run ``circuit`` once and return ``cirq.ResultDict``.

        Returned ``measurements[key]`` is a boolean array with shape
        ``(1, num_measured_qubits_for_key)``.
        """
        self.basis_states = [0]
        self.amplitudes = [1.0 + 0.0j]
        self.measurement_results = {}
        self._measurement_history = {}
        self._meas_len = {}
        # Circuit must use only qubits allocated using qubit manager of this simulator.
        context = cirq.DecompositionContext(self.qubit_manager)
        for op in circuit.all_operations():
            self._run_op(op, context)

        measurements = {
            key: np.asarray(
                [[((value >> i) & 1) == 1 for i in range(self._meas_len[key])]],
                dtype=np.bool_,
            )
            for key, value in self.measurement_results.items()
        }
        return cirq.ResultDict(params=cirq.ParamResolver({}), measurements=measurements)

    def read_register(self, register: Sequence[cirq.Qid]) -> int:
        """Read a little-endian register as an integer."""
        assert len(self.basis_states) == 1, (
            "Final state is superposition, add measurements."
        )
        s = self.basis_states[0]
        return sum(((s >> q.x) & 1) << i for i, q in enumerate(register))

    def _is_qubit_zero(self, qid: int) -> bool:
        """Checks whether qubit is in 0 state."""
        return all((s >> qid) & 1 == 0 for s in self.basis_states)