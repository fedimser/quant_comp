# MIT License, Dmytro Fedoriaka, 2021 (see LICENSE).

import numpy as np
import numba
from enum import Enum
import cirq


class BorderCondition(Enum):
    FIXED = 0     # Assume zeros outside the array.
    PERIODIC = 1  # Assume that leftmost and rightmost cell are neighbors.


@numba.jit("u8(u8,u8)")
def _get_bit(n, i):
    return (n >> i) % 2


@numba.jit("u8(u8,u8,u8,u8)")
def _step_fast(rule, p, n, s):
    """Simulates one step of ECA, where input and output states are encoded as integer.

    :param rule ECA rule (Wolfram code) - integer in range 0..255.
    :param p Equals to 0 for FIXED border condiditon; 1 for PERIODIC.
    :param n Number of cells.
    :param s Initial state. Little endian encoded (0-th cell - least significant bit).
    """
    ans = 0
    for i in range(1, n - 1):
        ans += _get_bit(rule, 4 * _get_bit(s, i - 1) + 2 * _get_bit(s, i) + _get_bit(s, i + 1)) << i
    ans += _get_bit(rule, 4 * p * _get_bit(s, n - 1) + 2 * _get_bit(s, 0) + _get_bit(s, 1))
    ans += _get_bit(rule, 4 * _get_bit(s, n - 2) + 2 * _get_bit(s, n - 1) + p * _get_bit(s, 0)) << (
            n - 1)
    return ans


@numba.jit("u8[:](u8,u8,u8)")
def _get_explicit_state_transitions(rule, p, n):
    ans = np.zeros(2 ** n, dtype=np.uint64)
    for state0 in range(0, 2 ** n):
        ans[state0] = _step_fast(rule, p, n, state0)
    return ans


class ECA:
    """
    Elementary Cellular Automaton.

    Defined by rule (integer in range 0..255) and border condition (periodic or fixed).
    """

    def __init__(self, rule, bord_cond):
        assert rule >= 0 and rule < 256
        self.rule = rule
        self.bord_cond = bord_cond
        if bord_cond == BorderCondition.FIXED:
            self.p = 0
        elif bord_cond == BorderCondition.PERIODIC:
            self.p = 1
        else:
            raise ValueError('Unknown border condition')

    def step(self, row0):
        """Simulates one step of ECA."""
        row1 = np.zeros_like(row0)
        n = len(row0)
        for i in range(1, n - 1):
            row1[i] = _get_bit(self.rule, 4 * row0[i - 1] + 2 * row0[i] + row0[i + 1])
        row1[0] = _get_bit(self.rule, 4 * self.p * row0[n - 1] + 2 * row0[0] + row0[1])
        row1[n - 1] = _get_bit(self.rule, 4 * row0[n - 2] + 2 * row0[n - 1] + self.p * row0[0])
        return row1

    def simulate_n_steps(self, row, steps):
        for _ in range(steps):
            row = self.step(row)
        return row

    def visualize(self, row0, steps):
        """Simulates n steps of ECA.

        Returns matrix with all intermediary configurations.
        """
        rows = np.zeros((steps + 1, len(row0)), dtype=np.int8)
        rows[0, :] = row0
        for i in range(1, steps + 1):
            rows[i, :] = self.step(rows[i - 1, :])
        return rows

    def get_explicit_state_transitions(self, n):
        """Computates 1-step transitions for all possible initial states.

        Returns list p of length 2**n, where p[i] equals to state to which
        this ECA transforms state i in 1 step. Both i an p[i] are integer
        (little endian) encodings of states.
        """
        assert n <= 20
        return _get_explicit_state_transitions(self.rule, self.p, n)


def _left_shift(qubits):
    circuit = cirq.Circuit()
    for i in range(len(qubits) - 1):
        circuit += cirq.SWAP(qubits[i], qubits[i + 1])
    return circuit


def _right_shift(qubits):
    circuit = cirq.Circuit()
    for i in range(len(qubits) - 2, -1, -1):
        circuit += cirq.SWAP(qubits[i], qubits[i + 1])
    return circuit


def circuit_for_eca(ca: ECA, qubits):
    """Implements circuit equivalent to one step of given Elementary Cellular Automaton."""
    assert 0 <= ca.rule <= 255, "Invalid rule."
    n = len(qubits)
    assert n >= 3, "Too few qubits."
    circuit = cirq.Circuit()

    if ca.rule == 204:
        # This rule leaves all cells unchanged.
        for i in range(n):
            circuit += cirq.I(qubits[i])
    elif ca.rule == 170:
        # Left shift.
        assert ca.bord_cond == BorderCondition.PERIODIC, "Rule requires periodic border condition."
        circuit += _left_shift(qubits)
    elif ca.rule == 240:
        # Right shift.
        assert ca.bord_cond == BorderCondition.PERIODIC, "Rule requires periodic border condition."
        circuit += _right_shift(qubits)
    elif ca.rule == 60:
        # This rule XORs all cells with its left neighbor.
        assert ca.bord_cond == BorderCondition.FIXED, "Rule requires fixed border condition."
        for i in range(n - 2, -1, -1):
            circuit += cirq.CNOT(qubits[i], qubits[i + 1])
    elif ca.rule == 102:
        # This rule XORs all cells with its right neighbor.
        assert ca.bord_cond == BorderCondition.FIXED, "Rule requires fixed border condition."
        for i in range(n - 1):
            circuit += cirq.CNOT(qubits[i + 1], qubits[i])
    elif ca.rule == 90:
        # This rule replaces cell with XOR of its left and right neigbors.
        assert n % 2 == 0, "Rule requires even n."
        assert ca.bord_cond == BorderCondition.FIXED, "Rule requires fixed border condition."
        for i in range(n - 3, -1, -1):
            circuit += cirq.CNOT(qubits[i], qubits[i + 2])
        for i in range(n - 2, 0, -2):
            circuit += cirq.CNOT(qubits[i], qubits[0])
        circuit += _left_shift(qubits)
    elif ca.rule == 150:
        # This rule XORs every cell with its left and right neighbor.
        if ca.bord_cond == BorderCondition.PERIODIC:
            assert n % 3 != 0, "Rule requires N%3!=0."
            for i in range(n - 1, 1, -1):
                circuit += cirq.CNOT(qubits[i - 2], qubits[i])
                circuit += cirq.CNOT(qubits[i - 1], qubits[i])
            if n % 3 == 2:
                circuit += cirq.SWAP(qubits[0], qubits[1])
            for i in range(n - 1, 0, -1):
                if (n - i) % 3 != 2 and not (i == 1 and n % 3 == 2):
                    circuit += cirq.CNOT(qubits[i], qubits[0])
            for i in range(n - 1, 1, -1):
                if (n - i) % 3 != 0:
                    circuit += cirq.CNOT(qubits[i], qubits[1])
            circuit += _left_shift(qubits)
        elif ca.bord_cond == BorderCondition.FIXED:
            assert n % 3 != 2, "Rule requires N%3!=2."
            for i in range(n - 1, 1, -1):
                circuit += cirq.CNOT(qubits[i - 2], qubits[i])
                circuit += cirq.CNOT(qubits[i - 1], qubits[i])
            circuit += cirq.CNOT(qubits[0], qubits[1])
            for i in range(n - 1, 0, -1):
                if (n - i) % 3 != 2:
                    circuit += cirq.CNOT(qubits[i], qubits[0])
            circuit += _left_shift(qubits)
    elif ca.rule == 166:
        assert n % 2 == 1, "This rule requires odd N."
        assert ca.bord_cond == BorderCondition.PERIODIC, "Rule requires periodic border condition."

        def c0mc1not(idx):
            n = len(idx)
            gate = cirq.X.controlled(n - 1, [0] + [1] * (n - 2))
            return gate(*[qubits[i] for i in idx])

        for i in range(n - 1, 1, -1):
            circuit += c0mc1not([i - 2, i - 1, i])
        for i in range(n // 2):
            controls = [2 * i + 2] + list(range(2 * i + 3, n, 2)) + [0, 1]
            circuit += c0mc1not(controls)
        for i in range(n // 2):
            controls = [2 * i + 1] + list(range(2 * i + 2, n, 2)) + [0]
            circuit += c0mc1not(controls)
        circuit += _left_shift(qubits)
    elif ca.rule == 154:
        circuit += [cirq.X(q) for q in qubits]
        circuit += circuit_for_eca(ECA(166, ca.bord_cond), qubits)
        circuit += [cirq.X(q) for q in qubits]
    elif ca.rule == 180:
        # Rule 180 is a mirror reflection of rule 166.
        return circuit_for_eca(ECA(166, ca.bord_cond), qubits[::-1])
    elif ca.rule == 210:
        # Rule 210 is a mirror reflection of rule 154.
        return circuit_for_eca(ECA(154, ca.bord_cond), qubits[::-1])
    elif ca.rule in [15, 45, 51, 75, 85, 89, 101, 105, 153, 165, 195]:
        # These rules can be implemented as applying other rule followed by flipping all states.
        circuit = circuit_for_eca(ECA(255 - ca.rule, ca.bord_cond), qubits)
        circuit += [cirq.X(q) for q in qubits]
    else:
        raise ValueError("Rule %d can't be implemented as quantum circuit." % ca.rule)
    return circuit
