"""Benchmarking: latency, memory, and structured-extraction accuracy.

Design goal is honest reporting over impressive-looking numbers:
  - Every result records which hardware tier and quantization path produced it.
  - If a tier can't be measured on the machine running this script (e.g. no CUDA
    device present), `run_hardware_benchmark` records it as `skipped` with a reason
    instead of inventing a number.
  - The GPT-4o Vision accuracy baseline only runs if `OPENAI_API_KEY` is set; if it
    isn't, `compare_to_gpt4o` returns a result with `ran=False` and a clear reason,
    rather than silently omitting the comparison or fabricating a score.
"""

from __future__ import annotations

import json
import os
import time
import tracemalloc
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import torch

from .engine import HardwareTier, ModelEngine, detect_hardware
from .extractors import extract_json


@dataclass
class LatencyResult:
    hardware: str
    quantization: str
    n_samples: int
    seconds_per_sample: list[float] = field(default_factory=list)
    mean_seconds: float = 0.0
    std_seconds: float = 0.0

    def finalize(self):
        import statistics

        self.mean_seconds = statistics.mean(self.seconds_per_sample)
        self.std_seconds = statistics.stdev(self.seconds_per_sample) if len(self.seconds_per_sample) > 1 else 0.0


@dataclass
class MemoryResult:
    hardware: str
    quantization: str
    peak_mb: float
    note: str = ""


@dataclass
class HardwareBenchmark:
    hardware: str
    skipped: bool
    reason: Optional[str] = None
    latency: Optional[LatencyResult] = None
    memory: Optional[MemoryResult] = None


@dataclass
class GPT4oComparison:
    ran: bool
    reason: Optional[str] = None
    n_samples: int = 0
    field_match_rate: Optional[float] = None
    per_sample: list[dict] = field(default_factory=list)


def _mps_peak_memory_mb() -> Optional[float]:
    if not torch.backends.mps.is_available():
        return None
    try:
        return torch.mps.driver_allocated_memory() / (1024 * 1024)
    except Exception:
        return None


def run_hardware_benchmark(
    image_paths: list[str],
    prompt: str,
    model_id: str,
    hardware: HardwareTier,
    n_repeats: int = 3,
) -> HardwareBenchmark:
    """Actually load the model and run generation on `hardware`. Returns a `skipped`
    result (with reason) rather than a fabricated number if the hardware isn't
    available on this machine.
    """
    if hardware == HardwareTier.CUDA and not torch.cuda.is_available():
        return HardwareBenchmark(hardware=hardware.value, skipped=True, reason="no CUDA device on this machine")
    if hardware == HardwareTier.MPS and not torch.backends.mps.is_available():
        return HardwareBenchmark(hardware=hardware.value, skipped=True, reason="MPS not available on this machine")

    from PIL import Image

    engine = ModelEngine(model_id=model_id, hardware=hardware, force_cpu=(hardware == HardwareTier.CPU))
    load_stats = engine.load()

    tracemalloc.start()
    latency = LatencyResult(hardware=hardware.value, quantization=load_stats.quantization, n_samples=0)

    for _ in range(n_repeats):
        for image_path in image_paths:
            img = Image.open(image_path).convert("RGB")
            start = time.time()
            engine.generate(img, prompt, max_new_tokens=256)
            latency.seconds_per_sample.append(time.time() - start)
            latency.n_samples += 1

    _, py_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    latency.finalize()

    mps_mb = _mps_peak_memory_mb()
    memory = MemoryResult(
        hardware=hardware.value,
        quantization=load_stats.quantization,
        peak_mb=mps_mb if mps_mb is not None else py_peak / (1024 * 1024),
        note="MPS driver allocation" if mps_mb is not None else "Python-tracked allocations only (not full process RSS)",
    )

    return HardwareBenchmark(hardware=hardware.value, skipped=False, latency=latency, memory=memory)


def compare_to_gpt4o(
    samples: list[dict],
    vf_generate,
) -> GPT4oComparison:
    """Compare VisionFlow's JSON extraction against GPT-4o Vision on the same images.

    `samples` is a list of {"image_path": ..., "prompt": ..., "schema": ...} dicts.
    `vf_generate` is a callable(image_path, prompt, schema) -> ExtractionResult from
    the local VisionFlow pipeline.

    Returns ran=False with a reason if OPENAI_API_KEY isn't set or the API call fails —
    never a fabricated accuracy number.
    """
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return GPT4oComparison(ran=False, reason="OPENAI_API_KEY not set; GPT-4o baseline not run")

    try:
        from openai import OpenAI
    except ImportError:
        return GPT4oComparison(ran=False, reason="openai package not installed (pip install visionflow[eval])")

    client = OpenAI(api_key=api_key)
    per_sample = []
    matches = 0
    total_fields = 0

    for sample in samples:
        image_path, prompt, schema = sample["image_path"], sample["prompt"], sample.get("schema")
        vf_result = vf_generate(image_path, prompt, schema)

        import base64

        with open(image_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()

        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": f"{prompt}\nRespond with a single JSON object only."},
                            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
                        ],
                    }
                ],
                max_tokens=768,
            )
            gpt4o_raw = response.choices[0].message.content
            gpt4o_result = extract_json(gpt4o_raw, generate_fn=None)
        except Exception as e:
            per_sample.append({"image_path": image_path, "error": f"GPT-4o API call failed: {e}"})
            continue

        vf_fields = vf_result.parsed if (vf_result and vf_result.ok) else {}
        gpt4o_fields = gpt4o_result.parsed if gpt4o_result.ok else {}

        sample_matches = 0
        sample_total = 0
        if isinstance(gpt4o_fields, dict):
            for key, gpt4o_value in gpt4o_fields.items():
                sample_total += 1
                total_fields += 1
                if isinstance(vf_fields, dict) and str(vf_fields.get(key, "")).strip().lower() == str(gpt4o_value).strip().lower():
                    sample_matches += 1
                    matches += 1

        per_sample.append(
            {
                "image_path": image_path,
                "visionflow_ok": bool(vf_result and vf_result.ok),
                "gpt4o_ok": gpt4o_result.ok,
                "field_matches": f"{sample_matches}/{sample_total}",
            }
        )

    field_match_rate = (matches / total_fields) if total_fields else None
    return GPT4oComparison(
        ran=True,
        n_samples=len(samples),
        field_match_rate=field_match_rate,
        per_sample=per_sample,
    )


def save_report(report: dict, path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(report, f, indent=2, default=lambda o: asdict(o) if hasattr(o, "__dataclass_fields__") else str(o))
