import numpy as np
import cirq

class BitFlipChannel:
    def __init__(self, flip_prob):
        self.flip_prob = flip_prob

    def transmit(self, ct, q):
        if np.random.rand() < self.flip_prob:
            ct.append(cirq.X(q))
        return q


class PhaseFlipChannel:
    def __init__(self, flip_prob):
        self.flip_prob = flip_prob

    def transmit(self, ct, q):
        if np.random.rand() < self.flip_prob:
            ct.append(cirq.Z(q))
        return q

class ArbitraryErrorChannel:
    def __init__(self, flip_prob):
        self.flip_prob = flip_prob
        
    def transmit(self, ct, q):
        if np.random.rand() < self.flip_prob:
            ct.append(cirq.Rx(rads=np.random.rand()*2*np.pi).on(q))
            ct.append(cirq.Ry(rads=np.random.rand()*2*np.pi).on(q))
            ct.append(cirq.Rz(rads=np.random.rand()*2*np.pi).on(q))
        return q