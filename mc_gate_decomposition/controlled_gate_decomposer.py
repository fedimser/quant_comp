import cirq
import numpy as np


def _unitary_power(U, p):
    """Raises unitary matrix U to power p."""
    assert cirq.is_unitary(U)
    eig_vals, eig_vectors = np.linalg.eig(U)
    Q = np.array(eig_vectors)
    return Q @ np.diag(np.exp(p * 1j * np.angle(eig_vals))) @ Q.conj().T


def _is_identity(M):
    return np.allclose(M, np.eye(M.shape[0]))


def _Ry(x):
    return np.array([[np.cos(x / 2), np.sin(x / 2)],
                     [-np.sin(x / 2), np.cos(x / 2)]])


def _Rz(x):
    return np.diag(np.exp([0.5j * x, -0.5j * x]))


def _flatten(x):
    return sum(x, [])


class ControlledGateDecomposer:
    """Decomposes single-qubit multi-controlled gate.

    Decomposition result is a sequence of single-qubit gates,
    CNOT gates and (optionally) CCNOT gates.

    === REFERENCE ===
    Barenco, Bennett et al. Elementary gates for quantum computation. 1995.
    https://arxiv.org/pdf/quant-ph/9503016.pdf
    """

    def __init__(self, allow_toffoli=True):
        """
        Args:
            allow_toffoli - whether Toffoli (CCNOT) gates are allowed in the
               output. If `False`, output will consist only of single-qubit
               gates and CNOT gates. Defaults to `True`.
        """
        self.allow_toffoli = allow_toffoli

    def _ccnot_true(self, c0, c1, target):
        """Implements action of Toffoli gate."""
        return [
            cirq.H.on(target),
            cirq.CNOT.on(c1, target),
            cirq.ZPowGate(exponent=-0.25).on(target),
            cirq.CNOT.on(c0, target),
            cirq.T.on(target),
            cirq.CNOT.on(c1, target),
            cirq.ZPowGate(exponent=-0.25).on(target),
            cirq.CNOT.on(c0, target),
            cirq.T.on(c1),
            cirq.T.on(target),
            cirq.H.on(target),
            cirq.CNOT.on(c0, c1),
            cirq.T.on(c0),
            cirq.ZPowGate(exponent=-0.25).on(c1),
            cirq.CNOT.on(c0, c1),
        ]

    def _ccnot_congruent(self, c0, c1, target):
        """Implements 'congruent' CCNOT.

        This is almost like CCNOT, but multiplies 4th qubit by -1.
        See lemma 6.2 in [1]."""
        return [
            cirq.ry(-np.pi / 4).on(target),
            cirq.CNOT(c1, target),
            cirq.ry(-np.pi / 4).on(target),
            cirq.CNOT(c0, target),
            cirq.ry(np.pi / 4).on(target),
            cirq.CNOT(c1, target),
            cirq.ry(np.pi / 4).on(target),
        ]

    def _ccnot(self, c0, c1, target, congruent=False):
        """Implements CCNOT gate.

        If `congruent=True`, implements "congruent" CCNOT gate, which differs
        from CCNOT in that it multiplies 4th qubit by -1, but takes fewer gates
        to implement.
        """
        if self.allow_toffoli:
            return [cirq.CCNOT.on(c0, c1, target)]
        elif congruent:
            return self._ccnot_congruent(c0, c1, target)
        else:
            return self._ccnot_true(c0, c1, target)

    def multi_ctrl_x(self, controls, target, free_qubits=[]):
        """Implements action of multi-controlled Pauli X gate .

        Result is guaranteed to consist exclusively of 1-qubit, CNOT and CCNOT
        gates.

        If `free_qubits` has at least 1 element, result has length
        O(len(controls)).

        Args:
            controls - control qubits. Can be empty.
            targets - target qubits.
            free_qubits - qubits which are neither controlled nor target. Can
                be modified by algorithm but will end up in inital state.
        """
        m = len(controls)
        if m == 0:
            return [cirq.X.on(target)]
        elif m == 1:
            return [cirq.CNOT.on(controls[0], target)]
        elif m == 2:
            return self._ccnot(controls[0], controls[1], target)

        m = len(controls)
        n = m + 1 + len(free_qubits)
        if (n >= 2 * m - 1) and (m >= 3):
            # See [1], Lemma 7.2.
            gates1 = [self._ccnot(controls[m - 2 - i],
                                  free_qubits[m - 4 - i],
                                  free_qubits[m - 3 - i],
                                  congruent=True) for i in range(m - 3)]
            gates2 = self._ccnot(
                controls[0],
                controls[1],
                free_qubits[0],
                congruent=True)
            gates3 = _flatten(gates1) + gates2 + _flatten(gates1[::-1])
            first_ccnot = self._ccnot(
                controls[m - 1], free_qubits[m - 3], target)
            return first_ccnot + gates3 + first_ccnot + gates3
        elif len(free_qubits) >= 1:
            # See [1], Lemma 7.3.
            m1 = n // 2
            free1 = controls[m1:] + [target] + free_qubits[1:]
            ctrl1 = controls[:m1]
            gates1 = self.multi_ctrl_x(
                ctrl1, free_qubits[0], free_qubits=free1)
            free2 = controls[:m1] + free_qubits[1:]
            ctrl2 = controls[m1:] + [free_qubits[0]]
            gates2 = self.multi_ctrl_x(ctrl2, target, free_qubits=free2)
            return gates1 + gates2 + gates1 + gates2
        else:
            # No free qubit - must use main algorithm.
            # This will never happen if called from main algorithm and is added
            # only for completeness.
            X = np.array([[0, 1], [1, 0]])
            return self.decompose(X, controls, target)

    def decompose(self, matrix, controls, target, free_qubits=[], power=1.0):
        """Implements action of multi-controlled unitary gate.

        Returns sequence of operations, which, when applied in sequence, gives
        result equivalent to applying single-qubit gate with matrix
        `matrix^power` on `target`, controlled by `controls`.

        Result is guaranteed to consist exclusively of 1-qubit, CNOT and CCNOT
        gates.

        If matrix is special unitary, result has length `O(len(controls))`.
        Otherwise result has length `O(len(controls)**2)`.

        Args:
            matrix - 2x2 numpy unitary matrix (of real or complex dtype).
            controls - control qubits.
            targets - target qubits.
            free_qubits - qubits which are neither controlled nor target. Can
                be modified by algorithm but will end up in inital state.
            power - power in which `unitary` should be raised. Deafults to 1.0.
        """
        assert cirq.is_unitary(matrix)
        assert matrix.shape == (2, 2)
        u = _unitary_power(matrix, power)

        # Matrix parameters, see definitions in [1], chapter 4.
        delta = np.angle(np.linalg.det(u)) * 0.5
        alpha = np.angle(u[0, 0]) + np.angle(u[0, 1]) - 2 * delta
        beta = np.angle(u[0, 0]) - np.angle(u[0, 1])
        theta = 2 * np.arccos(np.minimum(1.0, np.abs(u[0, 0])))

        # Decomposing matrix into three matrices - see [1], lemma 4.3.
        A = _Rz(alpha) @ _Ry(theta / 2)
        B = _Ry(-theta / 2) @ _Rz(-(alpha + beta) / 2)
        C = _Rz((beta - alpha) / 2)
        X = np.array([[0, 1], [1, 0]])
        if not np.allclose(A @ B @ C, np.eye(2), atol=1e-4):
            print('alpha', alpha)
            print('beta', beta)
            print('theta', theta)
            print('delta', delta)
            print('A', A)
            print('B', B)
            print('C', C)
            print(np.abs(A @ B @ C))
        assert np.allclose(A @ B @ C, np.eye(2), atol=1e-2)
        assert np.allclose(
            A @ X @ B @ X @ C,
            u /
            np.exp(
                1j *
                delta),
            atol=1e-2)

        m = len(controls)
        ctrl = controls
        assert m > 0, "No control qubits."

        if m == 1:
            # See [1], chapter 5.1.
            result = [
                cirq.ZPowGate(exponent=delta / np.pi).on(ctrl[0]),
                cirq.rz(-0.5 * (beta - alpha)).on(target),
                cirq.CNOT.on(ctrl[0], target),
                cirq.rz(0.5 * (beta + alpha)).on(target),
                cirq.ry(0.5 * theta).on(target),
                cirq.CNOT.on(ctrl[0], target),
                cirq.ry(-0.5 * theta).on(target),
                cirq.rz(-alpha).on(target),
            ]

            # Remove no-ops.
            result = [g for g in result if not _is_identity(g._unitary_())]

            return result
        else:
            gate_is_special_unitary = np.allclose(delta, 0)

            if gate_is_special_unitary:
                # O(n) decomposition of SU matrix - [1], lemma 7.9.
                cnot_seq = self.multi_ctrl_x(
                    ctrl[:-1], target, free_qubits=[ctrl[-1]])
                result = []
                result += self.decompose(C, [ctrl[-1]], target)
                result += cnot_seq
                result += self.decompose(B, [ctrl[-1]], target)
                result += cnot_seq
                result += self.decompose(A, [ctrl[-1]], target)
                return result
            else:
                # O(n^2) decomposition - [1], lemma 7.5.
                cnot_seq = self.multi_ctrl_x(
                    ctrl[:-1], ctrl[-1], free_qubits=free_qubits + [target])
                part1 = self.decompose(
                    matrix, [ctrl[-1]], target, power=0.5 * power)
                part2 = self.decompose(
                    matrix, [ctrl[-1]], target, power=-0.5 * power)
                part3 = self.decompose(matrix,
                                       ctrl[:-1],
                                       target,
                                       power=0.5 * power,
                                       free_qubits=free_qubits + [ctrl[-1]])
                return part1 + cnot_seq + part2 + cnot_seq + part3

        return circuit
