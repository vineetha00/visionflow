from visionflow.engine import HardwareTier, detect_hardware


def test_detect_hardware_returns_valid_tier():
    tier = detect_hardware()
    assert tier in (HardwareTier.CUDA, HardwareTier.MPS, HardwareTier.CPU)


def test_hardware_tier_values():
    assert HardwareTier.CUDA.value == "cuda"
    assert HardwareTier.MPS.value == "mps"
    assert HardwareTier.CPU.value == "cpu"
