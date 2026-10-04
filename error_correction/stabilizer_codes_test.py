from error_correction.stabilizer_codes import FIVE_QUBIT_PERFECT_CODE


def test_five_qubit_perfect_code():
    code = FIVE_QUBIT_PERFECT_CODE
    assert code.signature() == "[[5,1,3]]"
