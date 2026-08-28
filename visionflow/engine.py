"""Model loading and hardware-aware quantization for local VLM inference."""

from __future__ import annotations

import time
from dataclasses import dataclass
from enum import Enum
from typing import Optional

import torch
from PIL import Image
try:
    from transformers import AutoModelForImageTextToText as AutoModelForVLM
except ImportError:  # older transformers versions
    from transformers import AutoModelForVision2Seq as AutoModelForVLM
from transformers import AutoProcessor


class HardwareTier(str, Enum):
    CUDA = "cuda"
    MPS = "mps"
    CPU = "cpu"


class ModelSize(str, Enum):
    """Capability tiers. A 2.25B VLM is unusable on a Raspberry Pi even at INT4, so
    model size is part of hardware auto-detection rather than a fixed constant."""

    SMALL = "small"    # SmolVLM-256M  — Pi / low-RAM CPU
    MEDIUM = "medium"  # SmolVLM-500M  — 8GB machines, CPU-only laptops
    LARGE = "large"    # SmolVLM-2.25B — M-series Macs, discrete GPUs


MODEL_TIERS: dict[ModelSize, str] = {
    ModelSize.SMALL: "HuggingFaceTB/SmolVLM-256M-Instruct",
    ModelSize.MEDIUM: "HuggingFaceTB/SmolVLM-500M-Instruct",
    ModelSize.LARGE: "HuggingFaceTB/SmolVLM-Instruct",
}

DEFAULT_MODEL = MODEL_TIERS[ModelSize.LARGE]


def detect_hardware() -> HardwareTier:
    """Auto-detect the best available backend on this machine."""
    if torch.cuda.is_available():
        return HardwareTier.CUDA
    if torch.backends.mps.is_available():
        return HardwareTier.MPS
    return HardwareTier.CPU


def total_memory_gb() -> Optional[float]:
    """Total system RAM in GB, or None if it can't be determined."""
    try:
        import psutil

        return psutil.virtual_memory().total / (1024**3)
    except Exception:
        pass
    try:
        import os

        return (os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")) / (1024**3)
    except Exception:
        return None


def _cuda_vram_gb() -> Optional[float]:
    try:
        return torch.cuda.get_device_properties(0).total_memory / (1024**3)
    except Exception:
        return None


def select_model_size(hardware: Optional[HardwareTier] = None) -> ModelSize:
    """Pick a capability tier for this device.

    The thresholds are deliberately conservative: the cost of picking too large a
    model on a small device is a machine that swaps or OOMs, while picking too small
    costs accuracy that the benchmark table makes visible.
    """
    hardware = hardware or detect_hardware()
    ram = total_memory_gb()

    if hardware == HardwareTier.CUDA:
        vram = _cuda_vram_gb()
        if vram is None or vram >= 6:
            return ModelSize.LARGE
        return ModelSize.MEDIUM if vram >= 3 else ModelSize.SMALL

    if hardware == HardwareTier.MPS:
        # Unified memory: weights and activations share the RAM budget.
        if ram is None or ram >= 16:
            return ModelSize.LARGE
        return ModelSize.MEDIUM if ram >= 8 else ModelSize.SMALL

    # CPU: fp32 inference is ~2.4x slower than MPS fp16 on the same box (see
    # benchmarks), so bias one tier smaller than RAM alone would suggest.
    if ram is not None and ram < 4:
        return ModelSize.SMALL
    return ModelSize.MEDIUM


def select_model(hardware: Optional[HardwareTier] = None) -> str:
    """HF model id appropriate for this device."""
    return MODEL_TIERS[select_model_size(hardware)]


@dataclass
class LoadStats:
    hardware: HardwareTier
    quantization: str
    load_seconds: float
    peak_memory_mb: Optional[float] = None


@dataclass
class GenerationStats:
    """Per-generation measurements the benchmark harness needs to compute
    tokens/sec rather than only wall-clock seconds-per-sample."""

    text: str
    new_tokens: int
    seconds: float

    @property
    def tokens_per_second(self) -> float:
        return self.new_tokens / self.seconds if self.seconds > 0 else 0.0


class ModelEngine:
    """Loads a VLM with the quantization path appropriate for the detected hardware.

    - CUDA: bitsandbytes load_in_4bit, falls back to load_in_8bit if 4bit init fails.
    - MPS (Apple Silicon): float16 weights on the `mps` device. bitsandbytes INT4 is not
      reliable on MPS as of this writing, so this path trades quantization for a
      still-small fp16 footprint (SmolVLM-2B fp16 is ~4.5GB).
    - CPU: falls back to fp32/bf16 on CPU. A GGUF/llama.cpp path (`quantize.py`) is
      provided for users who want true INT4 on CPU via llama-cpp-python.
    """

    def __init__(
        self,
        model_id: Optional[str] = None,
        hardware: Optional[HardwareTier] = None,
        force_cpu: bool = False,
        quant_mode: Optional[str] = None,
    ):
        self.hardware = HardwareTier.CPU if force_cpu else (hardware or detect_hardware())
        # quant_mode pins the quantization instead of taking the tier's default.
        # The benchmark harness needs this: without it, asking for INT8 on CUDA
        # would silently measure INT4, since _load_cuda tries 4-bit first.
        self.quant_mode = quant_mode
        # model_id=None means "pick the capability tier that fits this device" —
        # a 2.25B VLM on a Pi is unusable, so size is part of auto-detection.
        self.model_id = model_id or select_model(self.hardware)
        self.auto_selected_model = model_id is None
        self.model = None
        self.processor = None
        self.load_stats: Optional[LoadStats] = None
        self.last_logits_processor = None

    def load(self) -> LoadStats:
        start = time.time()
        self.processor = AutoProcessor.from_pretrained(self.model_id)

        if self.hardware == HardwareTier.CUDA:
            quantization = self._load_cuda()
        elif self.hardware == HardwareTier.MPS:
            quantization = self._load_mps()
        else:
            quantization = self._load_cpu()

        elapsed = time.time() - start
        self.load_stats = LoadStats(
            hardware=self.hardware,
            quantization=quantization,
            load_seconds=elapsed,
        )
        return self.load_stats

    def _load_cuda(self) -> str:
        from .quantize import bnb_config

        def _load(mode: str) -> None:
            self.model = AutoModelForVLM.from_pretrained(
                self.model_id, quantization_config=bnb_config(mode), device_map="auto"
            )

        if self.quant_mode == "int8":
            _load("int8")
            return "int8-bnb"
        if self.quant_mode == "fp16":
            self.model = AutoModelForVLM.from_pretrained(
                self.model_id, torch_dtype=torch.float16, device_map="auto"
            )
            return "fp16-cuda"

        try:
            _load("int4")
            return "int4-bnb-nf4"
        except Exception:
            # Falling back is fine, but the returned label must say so — a row
            # claiming INT4 while running INT8 would corrupt every comparison
            # drawn from the benchmark table.
            _load("int8")
            return "int8-bnb-fallback"

    def _load_mps(self) -> str:
        self.model = AutoModelForVLM.from_pretrained(
            self.model_id, torch_dtype=torch.float16
        ).to("mps")
        return "fp16-mps"

    def _load_cpu(self) -> str:
        self.model = AutoModelForVLM.from_pretrained(
            self.model_id, torch_dtype=torch.float32
        )
        return "fp32-cpu"

    def generate(self, image: Image.Image, prompt: str, max_new_tokens: int = 512) -> str:
        """Free-form generation. See `generate_with_stats` for the measured variant."""
        return self.generate_with_stats(image, prompt, max_new_tokens=max_new_tokens).text

    def generate_with_stats(
        self,
        image: Image.Image,
        prompt: str,
        max_new_tokens: int = 512,
        logits_processor=None,
    ) -> GenerationStats:
        """Generate and report token count + wall-clock time.

        The benchmark harness needs tokens/sec, not just seconds-per-sample: SmolVLM
        emits wildly different output lengths per image category (a chart description
        runs 4-5x longer than a key-value form), so raw latency alone conflates
        "this backend is slow" with "this image produced more tokens".
        """
        if self.model is None:
            raise RuntimeError("Call .load() before .generate()")

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        text_prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = self.processor(text=text_prompt, images=[image], return_tensors="pt")

        device = self.hardware.value if self.hardware != HardwareTier.CPU else "cpu"
        inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}

        gen_kwargs = {"max_new_tokens": max_new_tokens}
        if logits_processor is not None:
            # Accept either a ready LogitsProcessorList or a factory taking the
            # prompt length — grammar processors need to know where the prompt
            # ends to reconstruct only the generated text, and only the engine
            # knows that after tokenization.
            if callable(logits_processor) and not hasattr(logits_processor, "__len__"):
                logits_processor = logits_processor(inputs["input_ids"].shape[1])
            if logits_processor is not None:
                gen_kwargs["logits_processor"] = logits_processor
                self.last_logits_processor = logits_processor

        start = time.perf_counter()
        with torch.no_grad():
            output_ids = self.model.generate(**inputs, **gen_kwargs)
        self._synchronize()
        elapsed = time.perf_counter() - start

        # Slice off the input tokens so only the newly generated continuation is
        # decoded — decoding the full sequence and string-matching the prompt back
        # off is unreliable because chat templates insert per-image placeholder
        # tokens that don't always round-trip identically through decode/encode.
        input_len = inputs["input_ids"].shape[1]
        new_tokens = output_ids[:, input_len:]
        generated = self.processor.batch_decode(new_tokens, skip_special_tokens=True)[0]
        return GenerationStats(
            text=generated.strip(),
            new_tokens=int(new_tokens.shape[1]),
            seconds=elapsed,
        )

    def _synchronize(self):
        """Block until queued device work finishes, so timings measure compute rather
        than kernel-launch time. MPS and CUDA both dispatch asynchronously."""
        try:
            if self.hardware == HardwareTier.CUDA:
                torch.cuda.synchronize()
            elif self.hardware == HardwareTier.MPS:
                torch.mps.synchronize()
        except Exception:
            pass
