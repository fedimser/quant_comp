# Sparse simulator for Cirq

This is a simple implementation of a Sparse simulator for circuits in Cirq.

It is sparse in a sense that it maintains a list of basis states with only 
non-zero amplitudes.

It supports unitary gates and computational basis measurments, and designed 
primarily for simulating arithmetic circuits.

I was not able to find such a simulator in Cirq (maybe I didn't look had 
enough), so I just wrote this one for my needs.

AI was used while writing this simulator.

## Usage

```python
import cirq

from cirq_sparse_sim.sparse_sim import SparseSimulator

simulator = SparseSimulator(seed=42)
q0, q1 = simulator.qubit_manager.qalloc(2)
circuit = cirq.Circuit(cirq.H(q0), cirq.CNOT(q0, q1), cirq.measure(q0, q1, key="m"))

samples = simulator.run(circuit, repetitions=100).measurements["m"]  # (100, 2)
trial = simulator.simulate(circuit)
wavefunction = trial.final_state_vector
steps = list(simulator.simulate_moment_steps(circuit))
```

The simulator implements `cirq.SimulatorBase` with sparse simulation states and
per-moment result snapshots. Each repetition starts in the all-zero state and
executes the same sparse gate and measurement logic. The simulator's
`basis_states`, `amplitudes`, `measurement_results`, and `read_register` expose
the last shot. Accessing a result's state vector explicitly materializes a dense
array; simulation itself remains sparse. Step-result sampling does not collapse
or otherwise change the saved state.

Qubits must still be allocated through `qubit_manager`. Nonempty parameter
resolvers and non-integer initial states are not supported. Integer initial
states and returned wavefunctions use Cirq's big-endian `qubit_order`;
`read_register` and sparse basis indices retain their little-endian convention.
For backwards compatibility, `run` accepts circuits without measurements and
returns only the last occurrence of each measurement key in each shot, rather
than all repeated-key records.

## Tests

From the repository root, with Cirq, NumPy, SymPy, and pytest installed:

```sh
python -m pytest cirq_sparse_sim/sparse_sim_test.py
```

The tests cover measurement results and final states, seeded random circuits
against Cirq's dense and Clifford simulators, independent repetitions, moment
steps and state snapshots, arithmetic gates, classical controls, reset,
decomposition, qubit management, and error cases. Dense state
comparisons retain global phase; Clifford comparisons allow an overall global
phase because stabilizer representations need not preserve it.