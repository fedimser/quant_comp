import time

import numpy as np
import cirq
import matplotlib.pyplot as plt

from cirq_sparse_sim.sparse_sim import SparseSimulator

from .channels import BitFlipChannel
from .protocols import QecProtocol


# Transmits random qubit using given channel and protocol and returns if transmission was successful.
# If output is passed, writes there circuit.
def test_protocol_once(protocol: QecProtocol, channel: cirq.Gate, output=None):
    # Generate qubit (in Bloch sphere notation).
    theta = np.random.rand() * np.pi
    phi = np.random.rand() * 2 * np.pi

    # Coordinates on Bloch sphere (for asserion in the end of experiment).
    init_bloch_coords = np.array(
        [np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)]
    )

    # Create circuit with one qubit and initialize it to generated state.
    ct = cirq.Circuit()
    q_init = cirq.NamedQubit("q_init")
    ct.append([cirq.ry(theta).on(q_init), cirq.rz(phi).on(q_init)])

    # Encode.
    encoded_qubits = protocol.encode(ct, q_init)

    # Transmit.
    transmitted_qubits = [channel.transmit(ct, q) for q in encoded_qubits]

    # Decode.
    decoded_qubit = protocol.decode(ct, transmitted_qubits)

    # Simulate cirquit to get final state of decoded qubit.
    sim = SparseSimulator()
    sim.simulate(ct)
    result = sim.simulate(ct)
    result_bloch_coords = result.bloch_vector_of(decoded_qubit)

    # Protocol should ensurre that decoded_qubit is not entangled with other qubits.
    if not np.allclose(np.linalg.norm(result_bloch_coords), 1.0):
        raise ValueError("Not pure state %s" % result_bloch_coords)

    if output != None:
        output["circuit"] = ct

    # Return whether qubit state was correctly transmitted.
    return np.linalg.norm(result_bloch_coords - init_bloch_coords) < 1e-5


# Experimentally calculates failure rate of error correcting protocol.
def test_protocol(protocol: QecProtocol, channel: cirq.Gate, num_experiments=100):
    ok_count = sum(
        [test_protocol_once(protocol, channel) for _ in range(num_experiments)]
    )
    return 1.0 - 1.0 * ok_count / num_experiments


def plot_errors(
    protocol,
    num_points=21,
    num_experiments=100,
    theoretical=None,
    channel_factory=lambda p: BitFlipChannel(p),
    max_p=1.0,
):
    time_start = time.time()
    channel_error = np.linspace(0, max_p, num_points)
    protocol_error = [
        test_protocol(protocol, channel_factory(p), num_experiments=num_experiments)
        for p in channel_error
    ]
    plt.plot(channel_error, protocol_error, label="Experiment")
    plt.xlabel("Channel error")
    plt.ylabel("Protocol error")
    plt.title(protocol.name)

    if not theoretical is None:
        plt.plot(channel_error, theoretical(channel_error), "--", label="Theory")
    plt.legend()
    plt.grid()
    plt.show()
    print(f"Time: {time.time() - time_start:.2f}s")
