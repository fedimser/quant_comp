from quantum_eca import circuit_for_eca, ECA, BorderCondition
import cirq
import numpy as np


def unitary_for_permutation(p):
    n = len(p)
    ans = np.zeros((n, n), dtype=np.complex128)
    for i in range(n):
        ans[p[i], i] = 1.0
    return ans


def verify(eca, N):
    """Checks that circuit implements unitary corresponding to correct permutation."""
    qubits = cirq.LineQubit.range(N)
    # Reverse qubits because of Cirq's qubit numbering convention.
    circuit = circuit_for_eca(eca, qubits[::-1])
    unitary_from_circuit = cirq.unitary(circuit)
    p = eca.get_explicit_state_transitions(N)
    unitary_from_ca = unitary_for_permutation(p)
    assert np.allclose(unitary_from_circuit, unitary_from_ca)


def verify_fixed(rule, ns):
    for n in ns:
        verify(ECA(rule, BorderCondition.FIXED), n)


def verify_periodic(rule, ns):
    for n in ns:
        verify(ECA(rule, BorderCondition.PERIODIC), n)


def test_crcuit_for_eca_all_correct():
    # For all eligible rules and values of N <= 10:
    #  - Generates circuit, calculates its unitary;
    #  - Generates permutation for ECA (classical), builds unitary for it.
    #  - Asserts that both unitaries are the same.
    verify_periodic(15, [3, 4, 5, 6, 7, 8, 9, 10])
    verify_periodic(45, [3, 5, 7, 9])
    verify_fixed(51, [3, 4, 5, 6, 7, 8, 9, 10])
    verify_periodic(51, [3, 4, 5, 6, 7, 8, 9, 10])
    verify_fixed(60, [3, 4, 5, 6, 7, 8, 9, 10])
    verify_periodic(75, [3, 5, 7, 9])
    verify_periodic(85, [3, 4, 5, 6, 7, 8, 9, 10])
    verify_periodic(89, [3, 5, 7, 9])
    verify_fixed(90, [4, 6, 8, 10])
    verify_periodic(101, [3, 5, 7, 9])
    verify_fixed(102, [4, 6, 8, 10])
    verify_fixed(105, [3, 4, 6, 7, 9, 10])
    verify_periodic(105, [4, 5, 7, 8, 10])
    verify_fixed(150, [3, 4, 6, 7, 9, 10])
    verify_periodic(150, [4, 5, 7, 8, 10])
    verify_fixed(153, [4, 5, 6, 7, 8, 9, 10])
    verify_periodic(154, [3, 5, 7, 9])
    verify_fixed(165, [4, 6, 8, 10])
    verify_periodic(166, [3, 5, 7, 9])
    verify_periodic(170, [3, 4, 5, 6, 7, 8, 9, 10])
    verify_periodic(180, [3, 5, 7, 9])
    verify_fixed(195, [3, 4, 5, 6, 7, 8, 9])
    verify_fixed(204, [3, 4, 5, 6, 7, 8, 9, 10])
    verify_periodic(204, [3, 4, 5, 6, 7, 8, 9, 10])
    verify_periodic(210, [3, 5, 7, 9])
    verify_periodic(240, [3, 4, 5, 6, 7, 8, 9, 10])
