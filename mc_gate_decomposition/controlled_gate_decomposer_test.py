import cirq
import numpy as np
from scipy.stats import unitary_group

from controlled_gate_decomposer import ControlledGateDecomposer

ccnot_matrix = cirq.CCNOT._unitary_()
cnot_matrix = cirq.CNOT._unitary_()


def _random_unitary():
    return unitary_group.rvs(2)


def _random_special_unitary():
    U = _random_unitary()
    return U / np.sqrt(np.linalg.det(U))


def _validate_matrix(u, allow_toffoli=False):
    if u.shape == (2, 2):
        pass
    elif u.shape == (4, 4):
        assert np.allclose(u, cnot_matrix)
    elif u.shape == (8, 8):
        assert allow_toffoli and np.allclose(u, ccnot_matrix)
    else:
        raise AssertionError('Bad matrix shape')


def _test_decomposition_with(U, m, dec):
    qubits = cirq.LineQubit.range(m + 1)

    gates = dec.decompose(U, qubits[:-1], qubits[-1])

    # Verify that all gates are either CNOT 1-qubit gates.
    for gate in gates:
        _validate_matrix(gate._unitary_(), dec.allow_toffoli)

    result_matrix = cirq.Circuit(gates).unitary()
    d = 2 ** (m + 1)
    expected_matrix = np.eye(d, dtype=np.complex128)
    expected_matrix[d - 2:d, d - 2:d] = U

    assert np.allclose(expected_matrix, result_matrix)


def _test_decomposition(U, m):
    """Test decomposition with given 2x2 matrix and number of controls."""
    _test_decomposition_with(U, m, ControlledGateDecomposer(allow_toffoli=True))
    _test_decomposition_with(U, m, ControlledGateDecomposer(allow_toffoli=False))


def test_specific_matrices():
    for gate in [cirq.X, cirq.Y, cirq.Z, cirq.H, cirq.I, cirq.T]:
        for m in range(1, 7):
            _test_decomposition(gate._unitary_(), m)


def test_unitary_matrices():
    for _ in range(10):
        U = unitary_group.rvs(2)
        for m in range(1, 6):
            _test_decomposition(U, m)


def test_special_unitary_matrices():
    for _ in range(10):
        U = _random_special_unitary()
        for m in range(1, 7):
            _test_decomposition(U, m)
