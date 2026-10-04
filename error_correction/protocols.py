import numpy as np
import cirq

from cirq import Circuit, Qid

from abc import ABC


class QecProtocol(ABC):
    def __init__(self):
        self.name = "No encoding"

    def encode(self, circuit: Circuit, qubit: Qid):
        return [qubit]

    def decode(self, circuit: Circuit, qubits: Qid):
        return qubits[0]

class NoEncodingProtocol(QecProtocol):
    def __init__(self):
        self.name = "No encoding"

    def encode(self, circuit: Circuit, qubit: Qid):
        return [qubit]

    def decode(self, circuit: Circuit, qubits: Qid):
        return qubits[0]


class ThreeQubitBitFlipProtocol(QecProtocol):
    def __init__(self):
        self.name = '3 qubit bit flip protocol'
    
    def encode(self, circuit, qubit):
        q0 = cirq.NamedQubit('aux_%d' % len(circuit.all_qubits()))
        circuit.append(cirq.CNOT(qubit, q0))
        q1 = cirq.NamedQubit('aux_%d' % len(circuit.all_qubits()))
        circuit.append(cirq.CNOT(qubit, q1))
        return [qubit, q0, q1]
    
    def decode(self, circuit, qubits):        
        def X(target):
            circuit.append(cirq.X(qubits[target]))
        
        def CCNOT(target):
            i1, i2 = 1, 2
            if target == 1:  i1, i2 = 0, 2
            if target == 2:  i1, i2 = 0, 1
            circuit.append(cirq.CCNOT(qubits[i1], qubits[i2], qubits[target]))
            
        X(1)
        CCNOT(0)
        X(1)
        CCNOT(1)
        CCNOT(0)
        X(0)
        CCNOT(2)
        X(0)
        X(2)
        CCNOT(0)
        X(0)
        X(2)
        CCNOT(2)
        X(0)
        X(1)
        CCNOT(0)
        X(1)
        CCNOT(1)
        CCNOT(0)
        
        # Measurement is not needed for Bit-Flip, but is needed so we can use it in Shor Code.
        circuit.append(cirq.measure(qubits[0]))
        circuit.append(cirq.measure(qubits[1]))
        
        return qubits[2]

class ThreeQubitPhaseFlipProtocol(QecProtocol):
    def __init__(self):
        self.name = '3 qubit phase flip protocol'
        self.bf_protocol = ThreeQubitBitFlipProtocol()
    
    def encode(self, circuit, qubit):
        qubits = self.bf_protocol.encode(circuit, qubit)
        for q in qubits:
            circuit.append(cirq.H(q))
        return qubits
    
    def decode(self, circuit, qubits):        
        for q in qubits:
            circuit.append(cirq.H(q))
        return self.bf_protocol.decode(circuit, qubits)



class ShorProtocol(QecProtocol):
    def __init__(self):
        self.name = 'Shor Code'
        self.bf_protocol = ThreeQubitBitFlipProtocol()
        self.pf_protocol = ThreeQubitPhaseFlipProtocol()
        
    
    def encode(self, circuit, qubit):
        result = []
        qubits1 = self.pf_protocol.encode(circuit, qubit)
        for q in qubits1:
            result += self.bf_protocol.encode(circuit, q)
        return result
    
    def decode(self, circuit, qubits):        
        return self.pf_protocol.decode(circuit, [
            self.bf_protocol.decode(circuit, qubits[0:3]),
            self.bf_protocol.decode(circuit, qubits[3:6]),
            self.bf_protocol.decode(circuit, qubits[6:9])
        ])


class NineQubitBitFlipProtocol(QecProtocol):
    def __init__(self):
        self.name = '9 qubit bit flip protocol'
        self.bf_protocol_1 = ThreeQubitBitFlipProtocol()
        self.bf_protocol_2 = ThreeQubitBitFlipProtocol()
    
    def encode(self, circuit, qubit):
        result = []
        qubits1 = self.bf_protocol_1.encode(circuit, qubit)
        for q in qubits1:
            result += self.bf_protocol_2.encode(circuit, q)
        return result
    
    def decode(self, circuit, qubits):        
        return self.bf_protocol_1.decode(circuit, [
            self.bf_protocol_2.decode(circuit, qubits[0:3]),
            self.bf_protocol_2.decode(circuit, qubits[3:6]),
            self.bf_protocol_2.decode(circuit, qubits[6:9])
        ])
        