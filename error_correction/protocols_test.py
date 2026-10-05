from error_correction.channels import ArbitraryErrorChannel
from error_correction.protocols import ShorProtocol
from error_correction.utils import test_protocol


def test_shor_code():
    protocol = ShorProtocol()
    channel = ArbitraryErrorChannel(0.01)
    assert test_protocol(protocol, channel, num_experiments=10) <= 0.1
