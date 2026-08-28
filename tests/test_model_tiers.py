"""Tests for device-aware model-size selection.

The point of these tiers is that a 2.25B VLM is unusable on a Raspberry Pi even at
INT4, so picking the model is part of hardware detection rather than a constant.
The failure that matters is picking *too large* — that OOMs or swaps a small device
— so these tests pin the upper bound at each tier.
"""

import pytest

from visionflow import engine
from visionflow.engine import MODEL_TIERS, HardwareTier, ModelSize, select_model, select_model_size


@pytest.fixture
def fake_ram(monkeypatch):
    def _set(gb):
        monkeypatch.setattr(engine, "total_memory_gb", lambda: gb)
    return _set


@pytest.mark.parametrize(
    "ram_gb,expected",
    [
        (2, ModelSize.SMALL),    # Raspberry Pi 4 / 2GB
        (4, ModelSize.SMALL),    # Pi 4 / 4GB
        (8, ModelSize.MEDIUM),   # M1 base
        (16, ModelSize.LARGE),   # M3 dev machine
        (64, ModelSize.LARGE),
    ],
)
def test_mps_tier_scales_with_unified_memory(fake_ram, ram_gb, expected):
    fake_ram(ram_gb)
    assert select_model_size(HardwareTier.MPS) is expected


@pytest.mark.parametrize("ram_gb,expected", [(2, ModelSize.SMALL), (16, ModelSize.MEDIUM)])
def test_cpu_biases_one_tier_smaller(fake_ram, ram_gb, expected):
    """CPU fp32 is several times slower than MPS fp16 on the same box, so a
    machine with plenty of RAM still shouldn't be handed the 2.25B model."""
    fake_ram(ram_gb)
    assert select_model_size(HardwareTier.CPU) is expected


@pytest.mark.parametrize(
    "vram_gb,expected",
    [(2, ModelSize.SMALL), (4, ModelSize.MEDIUM), (8, ModelSize.LARGE), (24, ModelSize.LARGE)],
)
def test_cuda_tier_scales_with_vram(monkeypatch, vram_gb, expected):
    monkeypatch.setattr(engine, "_cuda_vram_gb", lambda: vram_gb)
    assert select_model_size(HardwareTier.CUDA) is expected


def test_unknown_memory_falls_back_to_large_on_accelerators(fake_ram):
    """If RAM can't be probed, prefer capability on a GPU/MPS device rather than
    silently degrading accuracy on a machine that could have handled the big model."""
    fake_ram(None)
    assert select_model_size(HardwareTier.MPS) is ModelSize.LARGE


def test_every_tier_maps_to_a_model_id():
    for size in ModelSize:
        assert MODEL_TIERS[size].startswith("HuggingFaceTB/SmolVLM")
    assert len(set(MODEL_TIERS.values())) == len(ModelSize)


def test_select_model_returns_the_tier_model(fake_ram):
    fake_ram(2)
    assert select_model(HardwareTier.CPU) == MODEL_TIERS[ModelSize.SMALL]


def test_engine_records_when_the_model_was_auto_selected():
    auto = engine.ModelEngine(hardware=HardwareTier.CPU)
    pinned = engine.ModelEngine(model_id="HuggingFaceTB/SmolVLM-Instruct", hardware=HardwareTier.CPU)
    assert auto.auto_selected_model is True
    assert pinned.auto_selected_model is False
    assert pinned.model_id == "HuggingFaceTB/SmolVLM-Instruct"
