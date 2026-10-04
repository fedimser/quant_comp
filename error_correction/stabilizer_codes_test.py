import cirq
import numpy as np
import pytest

from error_correction.stabilizer_codes import (
    FIVE_QUBIT_PERFECT_CODE,
    SHOR_CODE,
    STEANE_CODE,
    SURFACE_17_CODE,
    StabilizerSet,
)


def test_five_qubit_perfect_code():
    code = FIVE_QUBIT_PERFECT_CODE
    assert code.signature() == "[[5,1,3]]"


def test_shor_code():
    code = SHOR_CODE
    assert code.signature() == "[[9,1,3]]"


def test_steane_code():
    code = STEANE_CODE
    assert code.signature() == "[[7,1,3]]"


def test_surface_17_code():
    code = SURFACE_17_CODE
    assert code.signature() == "[[9,1,3]]"


def test_stabilizer_set_rejects_noncommuting_generators():
    with pytest.raises(
        ValueError,
        match=r"Stabilizer generators 'XI' and 'ZI' must commute",
    ):
        StabilizerSet(["XI", "ZI"])


def test_stabilizer_set_rejects_dependent_generators():
    with pytest.raises(
        ValueError,
        match="Stabilizer generators must be independent",
    ):
        StabilizerSet(["XX", "XX"])


def test_five_qubit_perfect_code_corrects_single_qubit_errors():
    theta = 0.731
    phi = -0.294
    expected_bloch_vector = np.array(
        [
            np.sin(theta) * np.cos(phi),
            np.sin(theta) * np.sin(phi),
            np.cos(theta),
        ]
    )

    for qubit_index in range(5):
        for pauli in (cirq.X, cirq.Y, cirq.Z):
            circuit = cirq.Circuit()
            logical_qubit = cirq.NamedQubit("logical")
            circuit.append([cirq.ry(theta)(logical_qubit), cirq.rz(phi)(logical_qubit)])
            encoded_qubits = FIVE_QUBIT_PERFECT_CODE.encode(circuit, [logical_qubit])
            circuit.append(pauli(encoded_qubits[qubit_index]))
            decoded_qubit = FIVE_QUBIT_PERFECT_CODE.decode(circuit, encoded_qubits)[0]

            result = cirq.DensityMatrixSimulator(seed=1).simulate(circuit)
            num_qubits = len(result.qubit_map)
            density_matrix = cirq.partial_trace(
                result.final_density_matrix.reshape([2] * (2 * num_qubits)),
                [result.qubit_map[decoded_qubit]],
            )
            actual_bloch_vector = np.array(
                [
                    np.trace(density_matrix @ cirq.unitary(axis)).real
                    for axis in (cirq.X, cirq.Y, cirq.Z)
                ]
            )
            np.testing.assert_allclose(
                actual_bloch_vector, expected_bloch_vector, atol=1e-5
            )
