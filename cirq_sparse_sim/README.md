# Sparse simulator for Cirq

This is a simple implementation of a Sparse simulator for circuits in Cirq.

It is sparse in a sense that it maintains a list of basis states with only 
non-zero amplitudes.

It supports unitary gates and computational basis measurments, and designed 
primarily for simulating arithmetic circuits.

I was not able to find such a simulator in Cirq (maybe I didn't look had 
enough), so I just wrote this one for my needs.

AI was used while writing this simulator.

## Tests

From the repository root, with Cirq, NumPy, SymPy, and pytest installed:

```sh
python -m pytest cirq_sparse_sim/sparse_sim_test.py
```

The tests cover measurement results and final states, seeded random circuits
against Cirq's dense and Clifford simulators, arithmetic gates, classical
controls, reset, decomposition, qubit management, and error cases. Dense state
comparisons retain global phase; Clifford comparisons allow an overall global
phase because stabilizer representations need not preserve it.

This suite targets the existing single-shot `run(circuit)` API and its
`cirq.Result` output. Full `cirq.SimulatorBase` compatibility, including
`simulate`, repetitions, parameter sweeps, and arbitrary qubit types, is not
implemented or claimed by these tests.