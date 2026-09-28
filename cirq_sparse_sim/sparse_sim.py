"""Sparse SimulatorBase implementation for a restricted subset of Cirq circuits.

The simulator keeps the quantum state as a sparse list of computational basis
states and their amplitudes instead of a dense state vector. This is efficient
for circuits that mostly preserve sparsity (for example arithmetic/reversible
circuits with occasional single-qubit superposition gates).

Execution model:

1. Start in ``|0...0>`` represented as one basis state with amplitude ``1``.
2. Apply operations one by one, updating sparse basis states/amplitudes.
3. Measurements collapse the sparse state stochastically according to Born rule.
4. Repeat from the initial state for each requested shot.

This simulator is intentionally limited and aimed at tests and debugging.
Unsupported operations raise ``ValueError``.
"""

from __future__ import annotations

import copy
import math
import operator
import random
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from types import NotImplementedType
from typing import Any, Self

import cirq
import numpy as np
import sympy
from cirq.sim.simulator import check_all_resolved

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
    """Qubit manager used for legacy allocation and decomposition ancillas."""

    def __init__(self, simulator: SparseSimulator | _SparseState) -> None:
        """Initialize the simulator."""
        self.simulator = simulator
        self.num_qubits = 0
        self.free_qubits: list[int] = []
        self.free_qubits_set: set[int] = set()

    def _qid_for_id(self, qubit_id: int) -> cirq.Qid:
        if isinstance(self.simulator, _SparseState):
            return cirq.ops.CleanQubit(qubit_id, prefix="_sparse_sim")
        return cirq.LineQubit(qubit_id)

    @staticmethod
    def _qubit_id(qubit: cirq.Qid) -> int:
        if isinstance(qubit, cirq.LineQubit):
            return qubit.x
        if isinstance(qubit, cirq.ops.CleanQubit):
            return qubit.id
        raise AssertionError(f"Qubit {qubit} was not allocated by this manager")

    def _allocate_qubit(self) -> cirq.Qid:
        for index in range(len(self.free_qubits) - 1, -1, -1):
            qubit_id = self.free_qubits[index]
            qubit = self._qid_for_id(qubit_id)
            if not self.simulator._has_qubit(qubit):
                del self.free_qubits[index]
                self.free_qubits_set.remove(qubit_id)
                return qubit

        while True:
            qubit_id = self.num_qubits
            self.num_qubits += 1
            qubit = self._qid_for_id(qubit_id)
            if not self.simulator._has_qubit(qubit):
                return qubit
            self.free_qubits.append(qubit_id)
            self.free_qubits_set.add(qubit_id)

    def qalloc(self, n: int, dim: int = 2) -> list[cirq.Qid]:
        """Allocate qubits."""
        assert dim == 2
        qubits = [self._allocate_qubit() for _ in range(n)]
        for qubit in qubits:
            self.simulator._add_qubit(qubit)
        return qubits

    def qborrow(self, n: int, dim: int = 2) -> list[cirq.Qid]:
        """Not implemented."""
        raise NotImplementedError("qborrow is not supported")

    def qfree(self, qubits: Iterable[cirq.Qid]) -> None:
        """Free qubits."""
        for q in qubits:
            qubit_id = self._qubit_id(q)
            assert 0 <= qubit_id < self.num_qubits
            if qubit_id in self.free_qubits_set:
                raise RuntimeError(f"Qubit {qubit_id} released twice")
            if not self.simulator._has_qubit(q):
                raise RuntimeError(f"Qubit {q} is not in the simulation state")
            if not self.simulator._is_qubit_zero(q):
                raise RuntimeError(f"Qubit {qubit_id} released not in zero state")
            self.simulator._remove_qubit(q)
            self.free_qubits.append(qubit_id)
            self.free_qubits_set.add(qubit_id)

    def num_allocated_qubits(self) -> int:
        """Returns number of allocated qubits."""
        return self.num_qubits - len(self.free_qubits)

    def allocated_qubits(self) -> Iterator[cirq.LineQubit]:
        for qubit_id in range(self.num_qubits):
            if qubit_id not in self.free_qubits_set:
                yield cirq.LineQubit(qubit_id)

    def _copy_allocation_from(self, other: SparseSimQubitManager) -> None:
        self.num_qubits = other.num_qubits
        self.free_qubits = other.free_qubits.copy()
        self.free_qubits_set = other.free_qubits_set.copy()


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


class _SparseState(cirq.QuantumStateRepresentation, cirq.ClassicalDataStoreReader):
    """The sparse execution engine and its classical measurement history."""

    def __init__(
        self,
        random_source: random.Random,
        qubits: Sequence[cirq.Qid] = (),
        classical_data: cirq.ClassicalDataStoreReader | None = None,
    ) -> None:
        self.random: random.Random | np.random.RandomState = random_source
        self.axis_by_qubit = {qubit: axis for axis, qubit in enumerate(qubits)}
        self._next_axis = len(qubits)
        self._free_axes: list[int] = []
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
        self._records: dict[cirq.MeasurementKey, list[tuple[int, ...]]] = {}
        if classical_data is not None:
            if classical_data.channel_records:
                raise NotImplementedError(
                    "Channel measurement records are not supported"
                )
            for key, records in classical_data.records.items():
                self._records[key] = list(records)
                if records:
                    name = str(key)
                    self._measurement_history[name] = [
                        cirq.big_endian_bits_to_int(bits) for bits in records
                    ]
                    self._meas_len[name] = len(records[-1])
                    self.measurement_results[name] = sum(
                        bit << i for i, bit in enumerate(records[-1])
                    )

    def _has_qubit(self, qubit: cirq.Qid) -> bool:
        return qubit in self.axis_by_qubit

    def _add_qubit(self, qubit: cirq.Qid) -> None:
        if qubit in self.axis_by_qubit:
            raise RuntimeError(f"Qubit {qubit} is already in the simulation state")
        axis = self._free_axes.pop() if self._free_axes else self._next_axis
        if axis == self._next_axis:
            self._next_axis += 1
        self.axis_by_qubit[qubit] = axis

    def _remove_qubit(self, qubit: cirq.Qid) -> None:
        self._free_axes.append(self.axis_by_qubit.pop(qubit))

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
        self._records.setdefault(cirq.MeasurementKey.parse_serialized(key), []).append(
            tuple(measured_bits)
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
        qubit_ids = [self.axis_by_qubit[q] for q in op.qubits]

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

    def read_register(self, register: Sequence[cirq.Qid]) -> int:
        """Read a little-endian register as an integer."""
        assert (
            len(self.basis_states) == 1
        ), "Final state is superposition, add measurements."
        s = self.basis_states[0]
        try:
            return sum(
                ((s >> self.axis_by_qubit[q]) & 1) << i
                for i, q in enumerate(register)
            )
        except KeyError as ex:
            raise ValueError(
                f"Qubit {ex.args[0]} is not in this simulation state"
            ) from ex

    def _is_qubit_zero(self, qubit: cirq.Qid) -> bool:
        """Checks whether qubit is in 0 state."""
        axis = self.axis_by_qubit.get(qubit)
        return axis is None or all((s >> axis) & 1 == 0 for s in self.basis_states)

    def copy(self, deep_copy_buffers: bool = True) -> Self:
        """Copy the state; there are no reusable scratch buffers to share."""
        result = copy.copy(self)
        result.basis_states = self.basis_states.copy()
        result.amplitudes = self.amplitudes.copy()
        result.measurement_results = self.measurement_results.copy()
        result._measurement_history = {
            key: values.copy() for key, values in self._measurement_history.items()
        }
        result._meas_len = self._meas_len.copy()
        result._records = {key: values.copy() for key, values in self._records.items()}
        result.axis_by_qubit = self.axis_by_qubit.copy()
        result._next_axis = self._next_axis
        result._free_axes = self._free_axes.copy()
        result.qubit_manager = SparseSimQubitManager(result)
        result.qubit_manager._copy_allocation_from(self.qubit_manager)
        return result

    def measure(
        self, axes: Sequence[int], seed: cirq.RANDOM_STATE_OR_SEED_LIKE = None
    ) -> list[int]:
        if seed is not None:
            self.random = cirq.value.parse_random_state(seed)
        return [self._measure_qubit(axis) for axis in axes]

    def sample(
        self,
        axes: Sequence[int],
        repetitions: int = 1,
        seed: cirq.RANDOM_STATE_OR_SEED_LIKE = None,
    ) -> np.ndarray:
        repetitions = operator.index(repetitions)
        if repetitions < 0:
            raise ValueError("repetitions must be non-negative")
        return super().sample(axes, repetitions, seed).reshape(repetitions, len(axes))

    @property
    def records(self) -> Mapping[cirq.MeasurementKey, list[tuple[int, ...]]]:
        return self._records

    @property
    def channel_records(self) -> Mapping[cirq.MeasurementKey, list[int]]:
        return {}

    def keys(self) -> tuple[cirq.MeasurementKey, ...]:
        return tuple(self._records)

    def get_digits(self, key: cirq.MeasurementKey, index: int = -1) -> tuple[int, ...]:
        return self._records[key][index]

    def get_int(self, key: cirq.MeasurementKey, index: int = -1) -> int:
        return cirq.big_endian_bits_to_int(self.get_digits(key, index))


class _SparseSimulationState(cirq.SimulationState[_SparseState]):
    """Cirq's simulation-state interface over the existing sparse engine."""

    @property
    def sparse_state(self) -> _SparseState:
        return self._state

    @property
    def classical_data(self) -> cirq.ClassicalDataStoreReader:
        return self._state

    def get_axes(self, qubits: Sequence[cirq.Qid]) -> list[int]:
        for qubit in qubits:
            if qubit not in self.qubit_map:
                raise ValueError(f"Qubit {qubit} is not in this simulation state")
        try:
            return [self._state.axis_by_qubit[qubit] for qubit in qubits]
        except KeyError as ex:
            raise ValueError(
                f"Qubit {ex.args[0]} is not in this simulation state"
            ) from ex

    def apply_operation(self, op: cirq.Operation) -> None:
        self.get_axes(op.qubits)
        self._state._run_op(op, cirq.DecompositionContext(self._state.qubit_manager))

    def _act_on_fallback_(
        self, action: Any, qubits: Sequence[cirq.Qid], allow_decompose: bool = True
    ) -> bool | NotImplementedType:
        if isinstance(action, cirq.Gate):
            action = action.on(*qubits)
        if not isinstance(action, cirq.Operation):
            return NotImplemented
        self.apply_operation(action)
        return True

    def measure(
        self,
        qubits: Sequence[cirq.Qid],
        key: str,
        invert_mask: Sequence[bool],
        confusion_map: dict[tuple[int, ...], np.ndarray],
    ) -> None:
        self.apply_operation(
            cirq.measure(
                *qubits,
                key=key,
                invert_mask=tuple(invert_mask),
                confusion_map=confusion_map,
            )
        )

    def state_vector(self) -> np.ndarray:
        axes = self.get_axes(self.qubits)
        vector = np.zeros(1 << len(axes), dtype=np.complex128)
        included_bits = sum(1 << axis for axis in axes)
        for basis, amplitude in zip(
            self._state.basis_states, self._state.amplitudes, strict=True
        ):
            if basis & ~included_bits:
                raise ValueError("An omitted qubit is not in the zero state")
            index = 0
            for axis in axes:
                index = (index << 1) | ((basis >> axis) & 1)
            vector[index] = amplitude
        return vector


class SparseSimulatorStep(cirq.StepResultBase[_SparseSimulationState]):
    """A snapshot after a moment, with non-collapsing sampling support."""

    def state_vector(self, copy: bool = True) -> np.ndarray:
        """Materialize the sparse state in the simulation's qubit order."""
        return self._merged_sim_state.state_vector()


class SparseSimulatorTrialResult(
    cirq.SimulationTrialResultBase[_SparseSimulationState]
):
    """Final sparse simulation state, materialized as a vector only on request."""

    @property
    def final_state_vector(self) -> np.ndarray:
        return self._get_merged_sim_state().state_vector()


class SparseSimulator(
    cirq.SimulatorBase[
        SparseSimulatorStep, SparseSimulatorTrialResult, _SparseSimulationState
    ]
):
    """Sparse simulator for a restricted set of Cirq operations.

    Supports X, CNOT, CCNOT, numeric CZ powers, single-qubit unitaries,
    SparseSimArithmeticGate, measurement, reset, and classical controls using
    KeyCondition or SympyCondition. Other operations are tried by decomposition.

    ``run(circuit, repetitions=n)`` executes independent shots starting in zero.
    ``simulate`` and ``simulate_moment_steps`` expose final and intermediate
    states, with integer initial states interpreted in Cirq's qubit order.

    ``basis_states``, ``amplitudes``, and ``measurement_results`` describe the
    latest shot. Register reads and measurement_results are little-endian;
    Cirq state vectors and classical conditions use big-endian ordering.
    ``read_register`` requires a single basis state, so measure first if needed.

    Accepts circuits without measurements.
    """

    def __init__(self, seed: int | None = None) -> None:
        super().__init__(dtype=np.complex128, seed=seed, split_untangled_states=False)
        self.random = random.Random(seed)
        self._state = _SparseState(self.random)
        self.qubit_manager = SparseSimQubitManager(self)

    @property
    def basis_states(self) -> list[int]:
        return self._state.basis_states

    @property
    def amplitudes(self) -> list[complex]:
        return self._state.amplitudes

    @property
    def measurement_results(self) -> dict[str, int]:
        return self._state.measurement_results

    def read_register(self, register: Sequence[cirq.Qid]) -> int:
        """Read a little-endian register from the latest shot."""
        return self._state.read_register(register)

    def _has_qubit(self, qubit: cirq.Qid) -> bool:
        return self._state._has_qubit(qubit)

    def _add_qubit(self, qubit: cirq.Qid) -> None:
        self._state._add_qubit(qubit)

    def _remove_qubit(self, qubit: cirq.Qid) -> None:
        self._state._remove_qubit(qubit)

    def _is_qubit_zero(self, qubit: cirq.Qid) -> bool:
        return self._state._is_qubit_zero(qubit)

    def _create_partial_simulation_state(
        self,
        initial_state: Any,
        qubits: Sequence[cirq.Qid],
        classical_data: cirq.ClassicalDataStore,
    ) -> _SparseSimulationState:
        if not isinstance(initial_state, (int, np.integer)):
            raise NotImplementedError("Only integer initial states are supported")
        if not 0 <= initial_state < (1 << len(qubits)):
            raise ValueError("initial_state is out of range for the supplied qubits")
        for qubit in qubits:
            if qubit.dimension != 2:
                raise ValueError("Only dimension-2 qubits are supported")
        state_qubits = list(qubits)
        circuit_qubits = set(qubits)
        state_qubits.extend(
            qubit
            for qubit in self.qubit_manager.allocated_qubits()
            if qubit not in circuit_qubits
        )
        state = _SparseState(self.random, state_qubits, classical_data)
        basis = 0
        for i, qubit in enumerate(qubits):
            axis = state.axis_by_qubit[qubit]
            basis |= ((int(initial_state) >> (len(qubits) - i - 1)) & 1) << axis
        state.basis_states = [basis]
        state.qubit_manager._copy_allocation_from(self.qubit_manager)
        self._state = state
        return _SparseSimulationState(
            state=state, qubits=qubits, classical_data=classical_data, prng=self._prng
        )

    def _create_step_result(
        self, sim_state: cirq.SimulationStateBase[_SparseSimulationState]
    ) -> SparseSimulatorStep:
        self._state = sim_state.create_merged_state().sparse_state
        self.qubit_manager._copy_allocation_from(self._state.qubit_manager)
        return SparseSimulatorStep(sim_state.copy())

    def _create_simulator_trial_result(
        self,
        params: cirq.ParamResolver,
        measurements: dict[str, np.ndarray],
        final_simulator_state: cirq.SimulationStateBase[_SparseSimulationState],
    ) -> SparseSimulatorTrialResult:
        return SparseSimulatorTrialResult(params, measurements, final_simulator_state)

    def _can_be_in_run_prefix(self, val: Any) -> bool:
        # Preserve per-shot execution and the original decomposition context.
        return False

    def _core_iterator(
        self,
        circuit: cirq.AbstractCircuit,
        sim_state: cirq.SimulationStateBase[_SparseSimulationState],
        all_measurements_are_terminal: bool = False,
    ) -> Iterator[SparseSimulatorStep]:
        state = sim_state.create_merged_state()
        # Keep the engine's dispatch, including its supported classical controls.
        for moment in circuit if len(circuit) else [cirq.Moment()]:
            self._state = state.sparse_state
            try:
                for op in moment.operations:
                    if all_measurements_are_terminal and cirq.is_measurement(op):
                        continue
                    state.apply_operation(op)
            finally:
                self.qubit_manager._copy_allocation_from(self._state.qubit_manager)
            yield self._create_step_result(state)

    def simulate_sweep_iter(
        self,
        program: cirq.AbstractCircuit,
        params: cirq.Sweepable,
        qubit_order: cirq.QubitOrderOrList = cirq.QubitOrder.DEFAULT,
        initial_state: Any = None,
    ) -> Iterator[SparseSimulatorTrialResult]:
        resolvers = list(cirq.to_resolvers(params))
        for resolver in resolvers:
            check_all_resolved(cirq.resolve_parameters(program, resolver))
        return super().simulate_sweep_iter(
            program, resolvers, qubit_order, initial_state
        )

    def run_sweep_iter(
        self,
        program: cirq.AbstractCircuit,
        params: cirq.Sweepable,
        repetitions: int = 1,
    ) -> Iterator[cirq.Result]:
        for resolver in cirq.to_resolvers(params):
            yield cirq.ResultDict(
                params=resolver,
                records=self._run(program, resolver, repetitions),
            )

    def _run(
        self,
        circuit: cirq.AbstractCircuit,
        param_resolver: cirq.ParamResolver,
        repetitions: int,
    ) -> dict[str, np.ndarray]:
        circuit = cirq.resolve_parameters(circuit, param_resolver)
        check_all_resolved(circuit)
        repetitions = operator.index(repetitions)
        if repetitions < 0:
            raise ValueError("repetitions must be non-negative")
        operations = cirq.decompose(
            circuit,
            keep=lambda op: isinstance(op.gate, cirq.MeasurementGate)
            or not cirq.is_measurement(op),
        )
        measurement_widths: dict[str, list[int]] = {}
        for op in operations:
            if isinstance(op.gate, cirq.MeasurementGate):
                key = cirq.measurement_key_name(op)
                measurement_widths.setdefault(key, []).append(len(op.qubits))
        for key, widths in measurement_widths.items():
            if len(set(widths)) != 1:
                raise ValueError(
                    f"Different qid shapes for repeated measurement: key={key!r}"
                )

        if repetitions == 0:
            return {
                key: np.empty((0, len(widths), widths[0]), dtype=np.bool_)
                for key, widths in measurement_widths.items()
            }

        records: dict[str, list[Sequence[Sequence[int]]]] = {}
        qubits = tuple(sorted(circuit.all_qubits()))
        for _ in range(repetitions):
            for step in self._base_iterator(circuit, qubits, 0):
                pass
            for key, values in step._classical_data.records.items():
                records.setdefault(str(key), []).append(values)
        return {key: np.asarray(values, dtype=np.bool_) for key, values in records.items()}
