"""Tests for SparseSimulator's existing single-shot API."""

import itertools
import math
from collections.abc import Iterator, Sequence

import cirq
import numpy as np
import pytest
import sympy

from cirq_sparse_sim.sparse_sim import SparseSimArithmeticGate, SparseSimulator


def _state_vector(
    simulator: SparseSimulator, qubit_order: Sequence[cirq.Qid]
) -> np.ndarray:
    """Convert little-endian sparse indices to Cirq's specified tensor order."""
    qubit_ids = []
    for qubit in qubit_order:
        assert isinstance(qubit, cirq.LineQubit)
        qubit_ids.append(qubit.x)

    assert len(simulator.basis_states) == len(simulator.amplitudes)
    assert len(set(simulator.basis_states)) == len(simulator.basis_states)
    assert np.all(np.isfinite(simulator.amplitudes))
    np.testing.assert_allclose(
        np.linalg.norm(simulator.amplitudes), 1, rtol=0, atol=1e-12
    )

    vector = np.zeros(1 << len(qubit_ids), dtype=np.complex128)
    included_bits = sum(1 << qid for qid in qubit_ids)
    for basis, amplitude in zip(
        simulator.basis_states, simulator.amplitudes, strict=True
    ):
        assert basis >= 0
        assert basis & ~included_bits == 0, "An omitted qubit is not in |0>."
        index = 0
        for qid in qubit_ids:
            index = (index << 1) | ((basis >> qid) & 1)
        vector[index] = amplitude
    return vector


def _prepare_basis(
    qubits: Sequence[cirq.Qid], value: int
) -> list[cirq.Operation]:
    return [cirq.X(q) for i, q in enumerate(qubits) if (value >> i) & 1]


def _assert_matches_dense(
    simulator: SparseSimulator,
    circuit: cirq.Circuit,
    qubit_order: Sequence[cirq.Qid],
) -> cirq.Result:
    """Compare circuits with no measurements or only deterministic outcomes."""
    actual = simulator.run(circuit)
    expected = cirq.Simulator(dtype=np.complex128).simulate(
        circuit, qubit_order=qubit_order
    )
    np.testing.assert_allclose(
        _state_vector(simulator, qubit_order),
        expected.final_state_vector,
        rtol=0,
        atol=1e-8,
    )
    assert actual.measurements.keys() == expected.measurements.keys()
    for key, bits in expected.measurements.items():
        np.testing.assert_array_equal(actual.measurements[key], bits[np.newaxis, :])
    return actual


@pytest.mark.parametrize("num_qubits", [0, 1, 4])
def test_empty_circuit(num_qubits: int) -> None:
    simulator = SparseSimulator(seed=1)
    qubits = simulator.qubit_manager.qalloc(num_qubits)

    result = _assert_matches_dense(simulator, cirq.Circuit(), qubits)

    assert isinstance(result, cirq.ResultDict)
    assert result.params == cirq.ParamResolver({})
    assert result.measurements == {}
    assert simulator.measurement_results == {}
    assert simulator.basis_states == [0]
    assert simulator.amplitudes == [1]
    assert simulator.read_register(qubits) == 0
    assert simulator.read_register([]) == 0


@pytest.mark.parametrize("initial", range(8))
@pytest.mark.parametrize("gate", [cirq.X, cirq.CNOT, cirq.CCNOT], ids=repr)
def test_basis_permutations_measurements_and_final_state(
    initial: int, gate: cirq.Gate
) -> None:
    simulator = SparseSimulator()
    qubits = simulator.qubit_manager.qalloc(3)
    if gate == cirq.X:
        operation = gate(qubits[2])
        expected = initial ^ 4
    elif gate == cirq.CNOT:
        operation = gate(qubits[0], qubits[2])
        expected = initial ^ (4 if initial & 1 else 0)
    else:
        operation = gate(qubits[1], qubits[0], qubits[2])
        expected = initial ^ (4 if initial & 3 == 3 else 0)
    measurement_order = [qubits[2], qubits[0], qubits[1]]
    circuit = cirq.Circuit(
        _prepare_basis(qubits, initial),
        operation,
        cirq.measure(*measurement_order, key="result"),
    )

    result = _assert_matches_dense(simulator, circuit, qubits)

    expected_bits = [(expected >> i) & 1 for i in (2, 0, 1)]
    np.testing.assert_array_equal(result.measurements["result"], [expected_bits])
    assert simulator.basis_states == [expected]
    assert simulator.read_register(qubits) == expected
    assert simulator.read_register(qubits[::-1]) == cirq.big_endian_bits_to_int(
        [(expected >> i) & 1 for i in range(3)]
    )


@pytest.mark.parametrize("initial", [0, 1])
@pytest.mark.parametrize(
    "gate",
    [
        cirq.I,
        cirq.X,
        cirq.Y,
        cirq.Z,
        cirq.H,
        cirq.S,
        cirq.T,
        cirq.S**-1,
        cirq.T**-1,
        cirq.X**0.25,
        cirq.Y**-0.5,
        cirq.Z**1.25,
        cirq.rx(0.731),
        cirq.ry(-0.413),
        cirq.rz(1.271),
        cirq.rx(np.pi),
        cirq.XPowGate(exponent=1, global_shift=0.25),
        pytest.param(
            cirq.MatrixGate(cirq.testing.random_unitary(2, random_state=123)),
            id="random-unitary",
        ),
    ],
    ids=repr,
)
def test_single_qubit_unitaries(initial: int, gate: cirq.Gate) -> None:
    simulator = SparseSimulator()
    qubits = simulator.qubit_manager.qalloc(1)
    circuit = cirq.Circuit(_prepare_basis(qubits, initial), gate(*qubits))

    simulator.run(circuit)

    np.testing.assert_allclose(
        _state_vector(simulator, qubits),
        cirq.unitary(gate)[:, initial],
        rtol=0,
        atol=1e-9,
    )


@pytest.mark.parametrize(
    "gate",
    [
        cirq.CZ**0,
        cirq.CZ**0.25,
        cirq.CZ**-0.75,
        cirq.CZ**2,
        cirq.CZPowGate(exponent=0.5, global_shift=-0.5),
        cirq.CXPowGate(exponent=1, global_shift=-0.5),
        cirq.CCXPowGate(exponent=1, global_shift=0.25),
        cirq.CNOT**0.5,
        cirq.CCNOT**0.5,
        cirq.SWAP,
        cirq.ISWAP**0.5,
        cirq.CCZ**0.25,
        cirq.FSimGate(theta=0.3, phi=-0.7),
    ],
    ids=repr,
)
def test_multiqubit_gates_and_decompositions(gate: cirq.Gate) -> None:
    simulator = SparseSimulator()
    qubits = simulator.qubit_manager.qalloc(cirq.num_qubits(gate))
    circuit = cirq.Circuit(
        [cirq.ry(0.2 + 0.3 * i)(q) for i, q in enumerate(qubits)],
        cirq.T(qubits[0]),
        gate(*qubits[::-1]),
    )

    _assert_matches_dense(simulator, circuit, qubits)


@pytest.mark.parametrize("initial", [0, 1])
@pytest.mark.parametrize(
    "gate", cirq.SingleQubitCliffordGate.all_single_qubit_cliffords, ids=repr
)
def test_all_single_qubit_cliffords(initial: int, gate: cirq.Gate) -> None:
    simulator = SparseSimulator()
    qubits = simulator.qubit_manager.qalloc(1)

    _assert_matches_dense(
        simulator,
        cirq.Circuit(_prepare_basis(qubits, initial), gate(*qubits)),
        qubits,
    )


@pytest.mark.parametrize("seed", range(10))
@pytest.mark.parametrize("num_qubits", [1, 2, 3, 5])
def test_random_circuits_match_dense_simulator(seed: int, num_qubits: int) -> None:
    simulator = SparseSimulator(seed=seed)
    qubits = simulator.qubit_manager.qalloc(num_qubits)
    gate_domain = {
        cirq.X: 1,
        cirq.Y: 1,
        cirq.Z: 1,
        cirq.H: 1,
        cirq.S: 1,
        cirq.T: 1,
        cirq.X**0.37: 1,
        cirq.Y**-0.23: 1,
        cirq.rz(0.71): 1,
        cirq.MatrixGate(cirq.testing.random_unitary(2, random_state=seed)): 1,
        cirq.CNOT: 2,
        cirq.CZ**0.31: 2,
        cirq.SWAP: 2,
        cirq.ISWAP**0.5: 2,
        cirq.CCNOT: 3,
    }
    circuit = cirq.testing.random_circuit(
        qubits,
        n_moments=30,
        op_density=0.85,
        gate_domain=gate_domain,
        random_state=seed,
    )
    qubit_order = list(np.random.RandomState(seed).permutation(qubits))

    _assert_matches_dense(simulator, circuit, qubit_order)


@pytest.mark.parametrize("seed", range(10))
@pytest.mark.parametrize("num_qubits", [1, 2, 4, 6])
def test_random_clifford_circuits_match_clifford_simulator(
    seed: int, num_qubits: int
) -> None:
    simulator = SparseSimulator(seed=seed)
    qubits = simulator.qubit_manager.qalloc(num_qubits)
    gate_domain = {
        gate: 1 for gate in cirq.SingleQubitCliffordGate.all_single_qubit_cliffords
    }
    gate_domain.update({cirq.CNOT: 2, cirq.CZ: 2, cirq.SWAP: 2, cirq.ISWAP: 2})
    circuit = cirq.testing.random_circuit(
        qubits,
        n_moments=40,
        op_density=0.85,
        gate_domain=gate_domain,
        random_state=seed,
    )
    assert all(cirq.has_stabilizer_effect(op) for op in circuit.all_operations())
    qubit_order = qubits[::-1]

    simulator.run(circuit)
    expected = cirq.CliffordSimulator(seed=seed).simulate(
        circuit, qubit_order=qubit_order
    )

    # Stabilizer representations need not retain the circuit's global phase.
    cirq.testing.assert_allclose_up_to_global_phase(
        _state_vector(simulator, qubit_order),
        expected.final_state.state_vector(),
        atol=1e-7,
    )


@pytest.mark.parametrize("seed", range(5))
def test_random_circuit_followed_by_inverse_returns_to_zero(seed: int) -> None:
    simulator = SparseSimulator(seed=seed)
    qubits = simulator.qubit_manager.qalloc(4)
    circuit = cirq.testing.random_circuit(
        qubits, n_moments=25, op_density=0.9, random_state=seed
    )
    circuit += cirq.inverse(circuit)
    circuit += cirq.measure(*qubits, key="zero")

    result = _assert_matches_dense(simulator, circuit, qubits)

    np.testing.assert_array_equal(result.measurements["zero"], [[0, 0, 0, 0]])
    assert simulator.basis_states == [0]
    np.testing.assert_allclose(simulator.amplitudes, [1], atol=1e-9)


def test_destructive_interference_merges_and_prunes_states() -> None:
    simulator = SparseSimulator()
    qubits = simulator.qubit_manager.qalloc(3)
    circuit = cirq.Circuit(
        cirq.H.on_each(*qubits),
        cirq.Z(qubits[1]),
        cirq.H.on_each(*qubits),
    )

    _assert_matches_dense(simulator, circuit, qubits)

    assert simulator.basis_states == [2]
    np.testing.assert_allclose(simulator.amplitudes, [1], atol=1e-12)


def test_negligible_amplitudes_are_pruned_and_state_is_normalized() -> None:
    simulator = SparseSimulator()
    qubits = simulator.qubit_manager.qalloc(1)

    simulator.run(cirq.Circuit(cirq.ry(1e-10)(qubits[0])))

    assert simulator.basis_states == [0]
    np.testing.assert_allclose(_state_vector(simulator, qubits), [1, 0], atol=1e-12)


def test_small_but_significant_amplitudes_are_retained() -> None:
    simulator = SparseSimulator()
    qubits = simulator.qubit_manager.qalloc(1)
    gate = cirq.ry(1e-8)

    simulator.run(cirq.Circuit(gate(*qubits)))

    assert set(simulator.basis_states) == {0, 1}
    np.testing.assert_allclose(
        _state_vector(simulator, qubits), cirq.unitary(gate)[:, 0], rtol=0, atol=1e-12
    )


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
@pytest.mark.parametrize("num_qubits", [2, 3])
def test_entangled_measurements_and_collapsed_phase(
    seed: int, num_qubits: int
) -> None:
    simulator = SparseSimulator(seed=seed)
    qubits = simulator.qubit_manager.qalloc(num_qubits)
    circuit = cirq.Circuit(
        cirq.H(qubits[0]),
        [cirq.CNOT(qubits[0], q) for q in qubits[1:]],
        cirq.S(qubits[0]),
        cirq.measure(qubits[-1], key="first"),
        cirq.measure(*qubits, key="all"),
        cirq.measure(qubits[0], key="again"),
    )

    result = simulator.run(circuit)

    bit = int(result.measurements["first"][0, 0])
    assert bit in (0, 1)
    np.testing.assert_array_equal(result.measurements["all"], [[bit] * num_qubits])
    np.testing.assert_array_equal(result.measurements["again"], [[bit]])
    assert simulator.read_register(qubits) == bit * ((1 << num_qubits) - 1)
    expected = np.zeros(1 << num_qubits, dtype=np.complex128)
    expected[-1 if bit else 0] = 1j if bit else 1
    np.testing.assert_allclose(_state_vector(simulator, qubits), expected, atol=1e-12)


@pytest.mark.parametrize("seed", [0, 1])
def test_partial_measurement_preserves_unmeasured_superposition(seed: int) -> None:
    simulator = SparseSimulator(seed=seed)
    qubits = simulator.qubit_manager.qalloc(3)
    result = simulator.run(
        cirq.Circuit(
            cirq.H(qubits[0]),
            cirq.CNOT(qubits[0], qubits[1]),
            cirq.H(qubits[2]),
            cirq.measure(qubits[0], key="m"),
        )
    )

    bit = int(result.measurements["m"][0, 0])
    expected = np.zeros(8, dtype=np.complex128)
    expected[6 * bit : 6 * bit + 2] = 1 / math.sqrt(2)
    np.testing.assert_allclose(_state_vector(simulator, qubits), expected, atol=1e-12)
    with pytest.raises(AssertionError, match="superposition"):
        simulator.read_register(qubits)


@pytest.mark.parametrize("bit", [0, 1])
def test_deterministic_measurement_preserves_other_qubits(bit: int) -> None:
    simulator = SparseSimulator(seed=0)
    qubits = simulator.qubit_manager.qalloc(2)
    circuit = cirq.Circuit(
        _prepare_basis(qubits[:1], bit),
        cirq.H(qubits[1]),
        cirq.measure(qubits[0], key="m"),
    )

    result = _assert_matches_dense(simulator, circuit, qubits)

    np.testing.assert_array_equal(result.measurements["m"], [[bit]])
    assert len(simulator.basis_states) == 2


@pytest.mark.parametrize("initial", [0, 5, 7])
@pytest.mark.parametrize("invert_mask", [(), (True,), (True, False, True)])
def test_measurement_invert_mask_changes_result_not_state(
    initial: int, invert_mask: tuple[bool, ...]
) -> None:
    simulator = SparseSimulator()
    qubits = simulator.qubit_manager.qalloc(3)
    measurement_order = [qubits[2], qubits[0], qubits[1]]
    result = _assert_matches_dense(
        simulator,
        cirq.Circuit(
            _prepare_basis(qubits, initial),
            cirq.measure(*measurement_order, key="m", invert_mask=invert_mask),
        ),
        qubits,
    )

    bits = result.measurements["m"][0]
    assert simulator.basis_states == [initial]
    assert simulator.read_register(qubits) == initial
    assert simulator.measurement_results["m"] == sum(
        int(bit) << i for i, bit in enumerate(bits)
    )


@pytest.mark.parametrize("key", [None, "bits", cirq.MeasurementKey("bits", ("scope",))])
def test_measurement_keys_and_result_interface(
    key: str | cirq.MeasurementKey | None,
) -> None:
    simulator = SparseSimulator()
    qubits = simulator.qubit_manager.qalloc(3)
    measurement = cirq.measure(*qubits, key=key)
    name = cirq.measurement_key_name(measurement)
    result = simulator.run(
        cirq.Circuit(
            cirq.X(qubits[0]), measurement, cirq.measure(qubits[0], key="flag")
        )
    )

    assert isinstance(result, cirq.Result)
    assert result.params == cirq.ParamResolver({})
    assert result.measurements[name].shape == (1, 3)
    assert result.measurements[name].dtype == np.bool_
    np.testing.assert_array_equal(result.measurements[name], [[1, 0, 0]])
    np.testing.assert_array_equal(result.records[name], [[[1, 0, 0]]])
    assert result.histogram(key=name) == {4: 1}
    assert result.multi_measurement_histogram(keys=[name, "flag"]) == {(4, 1): 1}
    assert result.data[name].tolist() == [4]
    assert simulator.measurement_results[name] == 1
    assert simulator.read_register(qubits) == 1
    np.testing.assert_array_equal(_state_vector(simulator, qubits), np.eye(8)[4])


def test_repeated_measurement_key_retains_latest_value() -> None:
    simulator = SparseSimulator()
    qubits = simulator.qubit_manager.qalloc(2)
    result = simulator.run(
        cirq.Circuit(
            cirq.X(qubits[0]),
            cirq.measure(qubits[0], key="m"),
            cirq.X(qubits[0]),
            cirq.X(qubits[1]),
            cirq.measure(*qubits, key="m"),
        )
    )

    assert simulator.measurement_results == {"m": 2}
    np.testing.assert_array_equal(result.measurements["m"], [[0, 1]])
    assert simulator.read_register(qubits) == 2


def test_measurements_follow_nonuniform_born_probabilities() -> None:
    simulator = SparseSimulator(seed=12345)
    qubits = simulator.qubit_manager.qalloc(1)
    probability_one = 0.2
    trials = 1000
    circuit = cirq.Circuit(
        cirq.ry(2 * math.asin(math.sqrt(probability_one)))(qubits[0]),
        cirq.measure(qubits[0], key="m"),
    )
    ones = 0

    for _ in range(trials):
        result = simulator.run(circuit)
        bit = int(result.measurements["m"][0, 0])
        ones += bit
        assert simulator.read_register(qubits) == bit
        np.testing.assert_allclose(_state_vector(simulator, qubits), np.eye(2)[bit])

    standard_deviation = math.sqrt(trials * probability_one * (1 - probability_one))
    assert abs(ones - trials * probability_one) < 6 * standard_deviation


def test_seed_reproducibility_across_repeated_runs() -> None:
    outcomes = []
    for _ in range(2):
        simulator = SparseSimulator(seed=2468)
        qubits = simulator.qubit_manager.qalloc(2)
        circuit = cirq.Circuit(
            cirq.H.on_each(*qubits), cirq.measure(*qubits, key="m")
        )
        outcomes.append(
            [simulator.run(circuit).measurements["m"].tolist() for _ in range(32)]
        )

    assert outcomes[0] == outcomes[1]
    assert len({tuple(shot[0]) for shot in outcomes[0]}) > 1


def test_run_clears_previous_state_and_measurements_without_mutating_results() -> None:
    simulator = SparseSimulator(seed=0)
    qubits = simulator.qubit_manager.qalloc(2)
    first = simulator.run(
        cirq.Circuit(cirq.X.on_each(*qubits), cirq.measure(*qubits, key="old"))
    )

    second = simulator.run(cirq.Circuit(cirq.measure(qubits[0], key="new")))

    np.testing.assert_array_equal(first.measurements["old"], [[1, 1]])
    np.testing.assert_array_equal(second.measurements["new"], [[0]])
    assert set(second.measurements) == {"new"}
    assert simulator.measurement_results == {"new": 0}
    assert simulator.basis_states == [0]
    assert simulator.qubit_manager.num_allocated_qubits() == 2
    assert simulator.run(cirq.Circuit()).measurements == {}
    assert simulator.measurement_results == {}


@pytest.mark.parametrize("initial", [0, 1])
def test_reset_basis_state(initial: int) -> None:
    simulator = SparseSimulator()
    qubits = simulator.qubit_manager.qalloc(1)
    result = _assert_matches_dense(
        simulator,
        cirq.Circuit(
            _prepare_basis(qubits, initial),
            cirq.reset(qubits[0]),
            cirq.measure(qubits[0], key="m"),
        ),
        qubits,
    )

    np.testing.assert_array_equal(result.measurements["m"], [[0]])
    assert simulator.basis_states == [0]


@pytest.mark.parametrize("seed", [0, 1])
def test_reset_entangled_qubit_collapses_partner_without_recording_result(
    seed: int,
) -> None:
    simulator = SparseSimulator(seed=seed)
    qubits = simulator.qubit_manager.qalloc(2)
    result = simulator.run(
        cirq.Circuit(
            cirq.H(qubits[0]),
            cirq.CNOT(*qubits),
            cirq.reset(qubits[0]),
            cirq.measure(qubits[1], key="partner"),
        )
    )

    bit = int(result.measurements["partner"][0, 0])
    assert set(result.measurements) == {"partner"}
    assert simulator.measurement_results == {"partner": bit}
    assert simulator.read_register(qubits) == 2 * bit
    np.testing.assert_allclose(_state_vector(simulator, qubits), np.eye(4)[bit])
    simulator.qubit_manager.qfree(qubits[:1])


@pytest.mark.parametrize("bit", [0, 1])
@pytest.mark.parametrize("invert", [False, True])
def test_key_classical_control_uses_reported_measurement(
    bit: int, invert: bool
) -> None:
    simulator = SparseSimulator()
    qubits = simulator.qubit_manager.qalloc(2)
    circuit = cirq.Circuit(
        _prepare_basis(qubits[:1], bit),
        cirq.measure(qubits[0], key="control", invert_mask=(invert,)),
        cirq.X(qubits[1]).with_classical_controls("control"),
        cirq.measure(qubits[1], key="target"),
    )

    result = _assert_matches_dense(simulator, circuit, qubits)

    np.testing.assert_array_equal(result.measurements["target"], [[bit ^ invert]])
    assert simulator.read_register(qubits) == bit + 2 * (bit ^ invert)


@pytest.mark.parametrize("index", [-1, 0])
def test_key_condition_checks_entire_register_not_indexed_bit(index: int) -> None:
    simulator = SparseSimulator()
    qubits = simulator.qubit_manager.qalloc(3)
    condition = cirq.KeyCondition(cirq.MeasurementKey("control"), index=index)
    circuit = cirq.Circuit(
        cirq.X(qubits[1]),
        cirq.measure(*qubits[:2], key="control"),
        cirq.X(qubits[2]).with_classical_controls(condition),
        cirq.measure(qubits[2], key="target"),
    )

    result = _assert_matches_dense(simulator, circuit, qubits)

    np.testing.assert_array_equal(result.measurements["target"], [[1]])


@pytest.mark.parametrize("index, expected", [(0, 0), (1, 1), (-1, 1), (-2, 0)])
def test_key_condition_index_selects_measurement_occurrence(
    index: int, expected: int
) -> None:
    simulator = SparseSimulator()
    qubits = simulator.qubit_manager.qalloc(2)
    condition = cirq.KeyCondition(cirq.MeasurementKey("control"), index=index)
    result = simulator.run(
        cirq.Circuit(
            cirq.measure(qubits[0], key="control"),
            cirq.X(qubits[0]),
            cirq.measure(qubits[0], key="control"),
            cirq.X(qubits[1]).with_classical_controls(condition),
            cirq.measure(qubits[1], key="target"),
        )
    )

    np.testing.assert_array_equal(result.measurements["target"], [[expected]])
    assert simulator.read_register(qubits) == 1 + 2 * expected


@pytest.mark.parametrize("index", [1, -2])
def test_key_condition_rejects_nonexistent_measurement_occurrence(index: int) -> None:
    simulator = SparseSimulator()
    qubits = simulator.qubit_manager.qalloc(2)
    condition = cirq.KeyCondition(cirq.MeasurementKey("control"), index=index)

    with pytest.raises(IndexError):
        simulator.run(
            cirq.Circuit(
                cirq.measure(qubits[0], key="control"),
                cirq.X(qubits[1]).with_classical_controls(condition),
            )
        )


def test_classical_control_history_is_cleared_between_runs() -> None:
    simulator = SparseSimulator()
    qubits = simulator.qubit_manager.qalloc(2)
    simulator.run(
        cirq.Circuit(cirq.X(qubits[0]), cirq.measure(qubits[0], key="control"))
    )
    condition = cirq.KeyCondition(cirq.MeasurementKey("control"), index=0)
    circuit = cirq.Circuit(
        cirq.measure(qubits[0], key="control"),
        cirq.X(qubits[1]).with_classical_controls(condition),
        cirq.measure(qubits[1], key="target"),
    )

    result = _assert_matches_dense(simulator, circuit, qubits)

    np.testing.assert_array_equal(result.measurements["target"], [[0]])


@pytest.mark.parametrize("first, second", list(itertools.product([0, 1], repeat=2)))
def test_multiple_classical_controls_are_conjoined(first: int, second: int) -> None:
    simulator = SparseSimulator()
    qubits = simulator.qubit_manager.qalloc(3)
    circuit = cirq.Circuit(
        _prepare_basis(qubits, first + 2 * second),
        cirq.measure(qubits[0], key="a"),
        cirq.measure(qubits[1], key="b"),
        cirq.X(qubits[2]).with_classical_controls("a", "b"),
        cirq.measure(qubits[2], key="target"),
    )

    result = _assert_matches_dense(simulator, circuit, qubits)

    np.testing.assert_array_equal(result.measurements["target"], [[first & second]])


@pytest.mark.parametrize("initial", range(4))
@pytest.mark.parametrize(
    "expression",
    [
        sympy.Eq(sympy.Symbol("control"), 1),
        sympy.Eq(sympy.Symbol("control"), 2),
        sympy.Lt(sympy.Symbol("control"), 2),
        sympy.And(
            sympy.Ne(sympy.Symbol("control"), 0),
            sympy.Ne(sympy.Symbol("control"), 3),
        ),
    ],
)
def test_sympy_classical_conditions_use_cirq_integer_order(
    initial: int, expression: sympy.Basic
) -> None:
    simulator = SparseSimulator()
    qubits = simulator.qubit_manager.qalloc(3)
    circuit = cirq.Circuit(
        _prepare_basis(qubits[:2], initial),
        cirq.measure(*qubits[:2], key="control"),
        cirq.X(qubits[2]).with_classical_controls(cirq.SympyCondition(expression)),
        cirq.measure(qubits[2], key="target"),
    )

    _assert_matches_dense(simulator, circuit, qubits)


@pytest.mark.parametrize("initial", range(4))
def test_sympy_condition_substitutes_multiple_measurement_keys(initial: int) -> None:
    simulator = SparseSimulator()
    qubits = simulator.qubit_manager.qalloc(3)
    condition = cirq.SympyCondition(
        sympy.Eq(sympy.Symbol("a") + sympy.Symbol("b"), 1)
    )
    circuit = cirq.Circuit(
        _prepare_basis(qubits[:2], initial),
        cirq.measure(qubits[0], key="a"),
        cirq.measure(qubits[1], key="b"),
        cirq.X(qubits[2]).with_classical_controls(condition),
        cirq.measure(qubits[2], key="target"),
    )

    _assert_matches_dense(simulator, circuit, qubits)


@pytest.mark.parametrize("invert_mask", [(True, False), (False, True)])
def test_sympy_condition_uses_inverted_measurement(
    invert_mask: tuple[bool, ...],
) -> None:
    simulator = SparseSimulator()
    qubits = simulator.qubit_manager.qalloc(3)
    condition = cirq.SympyCondition(sympy.Eq(sympy.Symbol("control"), 2))
    circuit = cirq.Circuit(
        cirq.measure(*qubits[:2], key="control", invert_mask=invert_mask),
        cirq.X(qubits[2]).with_classical_controls(condition),
        cirq.measure(qubits[2], key="target"),
    )

    _assert_matches_dense(simulator, circuit, qubits)



@pytest.mark.parametrize("a, b", list(itertools.product(range(4), range(8))))
def test_arithmetic_registers_are_little_endian_and_preserve_spectators(
    a: int, b: int
) -> None:
    simulator = SparseSimulator()
    qubits = simulator.qubit_manager.qalloc(6)
    first = [qubits[2], qubits[0]]
    second = [qubits[4], qubits[1], qubits[3]]
    gate = SparseSimArithmeticGate([2, 3], lambda x, y: (x, (x + y) % 8))
    assert cirq.num_qubits(gate) == 5
    result = simulator.run(
        cirq.Circuit(
            _prepare_basis(first, a),
            _prepare_basis(second, b),
            cirq.X(qubits[5]),
            gate(*(first + second)),
            cirq.measure(*first, key="a"),
            cirq.measure(*second, key="b"),
            cirq.measure(qubits[5], key="spectator"),
        )
    )

    expected_b = (a + b) % 8
    assert simulator.read_register(first) == a
    assert simulator.read_register(second) == expected_b
    assert simulator.read_register(qubits[5:]) == 1
    np.testing.assert_array_equal(
        result.measurements["a"], [[(a >> i) & 1 for i in range(2)]]
    )
    np.testing.assert_array_equal(
        result.measurements["b"], [[(expected_b >> i) & 1 for i in range(3)]]
    )
    np.testing.assert_array_equal(result.measurements["spectator"], [[1]])
    expected_bits = [
        (a >> 1) & 1,
        (expected_b >> 1) & 1,
        a & 1,
        (expected_b >> 2) & 1,
        expected_b & 1,
        1,
    ]
    np.testing.assert_allclose(
        _state_vector(simulator, qubits),
        np.eye(64)[cirq.big_endian_bits_to_int(expected_bits)],
    )


def test_arithmetic_preserves_complex_superposition() -> None:
    simulator = SparseSimulator()
    qubits = simulator.qubit_manager.qalloc(4)
    preparation = cirq.Circuit(
        cirq.H.on_each(*qubits), cirq.T(qubits[0]), cirq.S(qubits[2])
    )
    gate = SparseSimArithmeticGate([2, 2], lambda a, b: (a, (a + b) % 4))
    permutation = np.zeros((16, 16), dtype=np.complex128)
    for a, b in itertools.product(range(4), repeat=2):
        permutation[4 * a + (a + b) % 4, 4 * a + b] = 1
    reference_circuit = preparation + cirq.Circuit(
        cirq.MatrixGate(permutation)(qubits[1], qubits[0], qubits[3], qubits[2])
    )

    simulator.run(preparation + cirq.Circuit(gate(*qubits)))
    reference = cirq.Simulator(dtype=np.complex128).simulate(
        reference_circuit, qubit_order=qubits
    )

    np.testing.assert_allclose(
        _state_vector(simulator, qubits), reference.final_state_vector, atol=1e-12
    )
    assert len(simulator.basis_states) == 16


def test_zero_width_arithmetic_registers() -> None:
    simulator = SparseSimulator()
    qubits = simulator.qubit_manager.qalloc(2)
    gate = SparseSimArithmeticGate(
        [0, 2], lambda empty, value: (empty, (value + 1) % 4)
    )
    zero_qubit_gate = SparseSimArithmeticGate([], lambda: ())

    simulator.run(
        cirq.Circuit(cirq.X.on_each(*qubits), gate(*qubits), zero_qubit_gate())
    )

    assert cirq.num_qubits(zero_qubit_gate) == 0
    assert simulator.read_register(qubits) == 0
    np.testing.assert_allclose(_state_vector(simulator, qubits), [1, 0, 0, 0])


def test_large_sparse_register_does_not_require_a_dense_state() -> None:
    simulator = SparseSimulator()
    qubits = simulator.qubit_manager.qalloc(128)
    gate = SparseSimArithmeticGate([128], lambda value: (value + 1,))
    expected = (1 << 127) + (1 << 63) + 1
    result = simulator.run(
        cirq.Circuit(
            cirq.X(qubits[127]),
            cirq.CNOT(qubits[127], qubits[63]),
            gate(*qubits),
            cirq.measure(qubits[127], qubits[63], qubits[0], key="set_bits"),
        )
    )

    assert simulator.basis_states == [expected]
    assert simulator.amplitudes == [1]
    assert simulator.read_register(qubits) == expected
    np.testing.assert_array_equal(result.measurements["set_bits"], [[1, 1, 1]])


class _AncillaPhaseGate(cirq.Gate):
    def __init__(self, *, release: bool = True, uncompute: bool = True) -> None:
        self.release = release
        self.uncompute = uncompute

    def _num_qubits_(self) -> int:
        return 1

    def _decompose_with_context_(
        self, qubits: Sequence[cirq.Qid], context: cirq.DecompositionContext
    ) -> Iterator[cirq.OP_TREE]:
        ancilla = context.qubit_manager.qalloc(1)[0]
        yield [cirq.CNOT(qubits[0], ancilla), [cirq.Z(ancilla)]]
        if self.uncompute:
            yield cirq.CNOT(qubits[0], ancilla)
        if self.release:
            context.qubit_manager.qfree([ancilla])


class _NestedPhaseGate(cirq.Gate):
    def _num_qubits_(self) -> int:
        return 1

    def _decompose_(self, qubits: Sequence[cirq.Qid]) -> Iterator[cirq.OP_TREE]:
        yield [[_AncillaPhaseGate()(qubits[0])]]


def test_recursive_decomposition_allocates_uncomputes_and_reuses_ancilla() -> None:
    simulator = SparseSimulator()
    qubits = simulator.qubit_manager.qalloc(1)
    circuit = cirq.Circuit(
        cirq.H(qubits[0]),
        _NestedPhaseGate()(qubits[0]),
        _NestedPhaseGate()(qubits[0]),
        cirq.H(qubits[0]),
        cirq.measure(qubits[0], key="m"),
    )

    result = simulator.run(circuit)

    np.testing.assert_array_equal(result.measurements["m"], [[0]])
    np.testing.assert_allclose(_state_vector(simulator, qubits), [1, 0], atol=1e-12)
    assert simulator.qubit_manager.num_allocated_qubits() == 1
    assert simulator.qubit_manager.num_qubits == 2
    assert simulator.qubit_manager.qalloc(1) == [cirq.LineQubit(1)]


def test_tagged_and_circuit_operations() -> None:
    simulator = SparseSimulator()
    qubits = simulator.qubit_manager.qalloc(2)
    subcircuit = cirq.FrozenCircuit(
        cirq.H(qubits[0]), cirq.CNOT(*qubits), cirq.S(qubits[1])
    )
    circuit = cirq.Circuit(
        cirq.CircuitOperation(subcircuit).repeat(3),
        cirq.X(qubits[0]).with_tags("tag"),
    )

    _assert_matches_dense(simulator, circuit, qubits)


def test_qubit_allocation_release_and_reuse() -> None:
    simulator = SparseSimulator()
    manager = simulator.qubit_manager
    assert isinstance(manager, cirq.QubitManager)
    assert manager.qalloc(0) == []
    assert manager.num_allocated_qubits() == 0
    qubits = manager.qalloc(4)
    assert qubits == cirq.LineQubit.range(4)
    assert manager.num_allocated_qubits() == 4

    manager.qfree([qubits[1], qubits[3]])

    assert manager.num_allocated_qubits() == 2
    assert manager.qalloc(3) == [qubits[3], qubits[1], cirq.LineQubit(4)]
    assert manager.num_allocated_qubits() == 5
    assert manager.free_qubits == []
    assert manager.free_qubits_set == set()
    manager.qfree([])
    assert manager.num_allocated_qubits() == 5


def test_reused_qubits_simulate_with_noncontiguous_register_order() -> None:
    simulator = SparseSimulator()
    original = simulator.qubit_manager.qalloc(6)
    simulator.qubit_manager.qfree([original[1], original[4]])
    recycled = simulator.qubit_manager.qalloc(2)
    qubits = [recycled[1], original[5], recycled[0]]
    circuit = cirq.Circuit(
        cirq.H(qubits[0]), cirq.CNOT(qubits[0], qubits[2]), cirq.Y(qubits[1])
    )

    _assert_matches_dense(simulator, circuit, qubits)


@pytest.mark.parametrize("gate", [cirq.X, cirq.H], ids=repr)
def test_qubits_cannot_be_freed_unless_every_branch_is_zero(gate: cirq.Gate) -> None:
    simulator = SparseSimulator()
    qubits = simulator.qubit_manager.qalloc(1)
    simulator.run(cirq.Circuit(gate(*qubits)))

    with pytest.raises(RuntimeError, match="not in zero state"):
        simulator.qubit_manager.qfree(qubits)

    assert simulator.qubit_manager.num_allocated_qubits() == 1
    assert simulator.qubit_manager.free_qubits_set == set()


def test_double_release_is_rejected() -> None:
    simulator = SparseSimulator()
    qubits = simulator.qubit_manager.qalloc(1)
    simulator.qubit_manager.qfree(qubits)

    with pytest.raises(RuntimeError, match="released twice"):
        simulator.qubit_manager.qfree(qubits)

    assert simulator.qubit_manager.num_allocated_qubits() == 0
    assert simulator.qubit_manager.qalloc(2) == cirq.LineQubit.range(2)


@pytest.mark.parametrize(
    "qubit", [cirq.NamedQubit("invalid"), cirq.LineQubit(-1), cirq.LineQubit(1)]
)
def test_free_rejects_invalid_qubits(qubit: cirq.Qid) -> None:
    simulator = SparseSimulator()
    simulator.qubit_manager.qalloc(1)

    with pytest.raises(AssertionError):
        simulator.qubit_manager.qfree([qubit])

    assert simulator.qubit_manager.num_allocated_qubits() == 1


def test_nonbinary_allocation_and_borrowing_are_unsupported() -> None:
    simulator = SparseSimulator()

    with pytest.raises(AssertionError):
        simulator.qubit_manager.qalloc(1, dim=3)
    with pytest.raises(NotImplementedError, match="qborrow is not supported"):
        simulator.qubit_manager.qborrow(1)

    assert simulator.qubit_manager.num_allocated_qubits() == 0


@pytest.mark.parametrize(
    "gate, message",
    [
        (_AncillaPhaseGate(release=False), "did not free all allocated qubits"),
        (_AncillaPhaseGate(uncompute=False), "not in zero state"),
    ],
)
def test_invalid_ancilla_decompositions_raise(gate: cirq.Gate, message: str) -> None:
    simulator = SparseSimulator()
    qubits = simulator.qubit_manager.qalloc(1)

    with pytest.raises(RuntimeError, match=message):
        simulator.run(cirq.Circuit(cirq.X(qubits[0]), gate(*qubits)))


class _UnsupportedGate(cirq.Gate):
    def _num_qubits_(self) -> int:
        return 2


@pytest.mark.parametrize(
    "gate",
    [
        _UnsupportedGate(),
        cirq.depolarize(0.1),
        cirq.amplitude_damp(0.1),
        cirq.X ** sympy.Symbol("theta"),
        cirq.CZ ** sympy.Symbol("theta"),
    ],
    ids=repr,
)
def test_unsupported_operations_raise_value_error(gate: cirq.Gate) -> None:
    simulator = SparseSimulator()
    qubits = simulator.qubit_manager.qalloc(cirq.num_qubits(gate))

    with pytest.raises(ValueError, match="Operation cannot be simulated"):
        simulator.run(cirq.Circuit(gate(*qubits)))


def test_unsupported_classical_condition_raises_value_error() -> None:
    simulator = SparseSimulator()
    qubits = simulator.qubit_manager.qalloc(1)
    condition = cirq.BitMaskKeyCondition(cirq.MeasurementKey("m"), bitmask=1)
    circuit = cirq.Circuit(
        cirq.measure(*qubits, key="m"),
        cirq.X(*qubits).with_classical_controls(condition),
    )

    with pytest.raises(ValueError, match="Unsupported classical condition"):
        simulator.run(circuit)


@pytest.mark.parametrize(
    "condition",
    ["missing", cirq.SympyCondition(sympy.Eq(sympy.Symbol("missing"), 1))],
)
def test_missing_classical_measurement_is_not_silently_ignored(
    condition: str | cirq.Condition,
) -> None:
    simulator = SparseSimulator()
    qubits = simulator.qubit_manager.qalloc(1)

    with pytest.raises((KeyError, ValueError), match="missing"):
        simulator.run(
            cirq.Circuit(cirq.X(*qubits).with_classical_controls(condition))
        )


@pytest.mark.parametrize("output", [(), (0, 0)])
def test_arithmetic_gate_rejects_wrong_output_register_count(
    output: tuple[int, ...],
) -> None:
    simulator = SparseSimulator()
    qubits = simulator.qubit_manager.qalloc(1)
    gate = SparseSimArithmeticGate([1], lambda _: output)

    with pytest.raises(AssertionError):
        simulator.run(cirq.Circuit(gate(*qubits)))