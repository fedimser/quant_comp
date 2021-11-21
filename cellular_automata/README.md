## Quantum Elementary Cellular Automata

This directory contains materials for my research project on Quantum Elementary Cellular Automata.

### Summary

[Elementary Cellular Automaton](https://en.wikipedia.org/wiki/Elementary_cellular_automaton) (ECA) is 1-dimensional
cellular automata. It consists of N cells in a row, each being in state 0 or 1 (i.e. a bit array). On every state
state of every cells is updated according to specified rules, depending on state of this cell and its two
neigbors on previous iteration. There are in total 256 different rules, defining different ECAs.

The idea is to replace bit array with array of [qubits](https://en.wikipedia.org/wiki/Qubit) and implement transition rule 
with a [Quantum circuit](https://en.wikipedia.org/wiki/Quantum_circuit), so we can have a quantum system which acts on basis vectors exactly as 
classical ECA acts on bit arrays.

The objective of this project is to find all rules for which it's in principle possible to find corresponding 
circuit, and for every such rule give instruction how to construct that circuit. Turns out that there are 22 such 
rules and we can efficinelty build a circuit for all of them.

### Contents

* quantum_eca.py - Python code implementing circuits for all reversible Elementary Cellular Automata.

* quantum_eca_test.py - tests verifying correctness of quantum circuits.

* [Finding all reversible rules](finding_all_reversible_rules.ipynb) - experiment used to identify which 
  ECA rules are reversible
  
* [Proving irreversibility](proving_irreversibility.ipynb) - in this notebook I show how to find 2 states which are mapped to the 
  same state by given rule, which can be used as proof that rule is not reversible.

* [Rule 90 in Cirq](rule90_cirq.ipynb) - example how we can simulate Rule 90 in Cirq.

* [Rule 90 in Q#](rule90_qsharp.ipynb) - Q# Notebook demonstrating how we can implement and simulate Rule 90 in Q#.

* [ECA pictures](eca_pictures.ipynb) - This notebook visualizes all 22 reversible ECAs.

### Paper

Results of this project are summarized in my paper 
"Quantum Circuits for Elementary Cellular Automata",
available [here](http://dx.doi.org/10.13140/RG.2.2.22346.08641).

### License

All files in this directory are licenced under MIT license, as well as rest of the repository. Please refer to LICENSE
file at root for more details.