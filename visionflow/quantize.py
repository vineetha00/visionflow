"""Standalone quantization helpers, usable outside of ModelEngine for
inspection or scripted conversion.

Three paths, matched to hardware:
  - CUDA:  bitsandbytes NF4 (int4) via BitsAndBytesConfig, int8 fallback.
  - MPS:   no reliable INT4 kernel path yet; fp16 weights on `mps` device
           (see engine.ModelEngine._load_mps). Included here for completeness
           and so the auto-detection logic has one place to live.
  - CPU:   GGUF quantization via llama.cpp for users who want true INT4/INT8
           on CPU. Requires `llama-cpp-python` and a GGUF-converted checkpoint
           (see `convert_to_gguf` docstring for the conversion path, which
           shells out to llama.cpp's own conversion script since there is no
           pure-Python GGUF quantizer for vision models yet).
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Literal

QuantMode = Literal["int4", "int8", "fp16"]


def bnb_config(mode: QuantMode = "int4"):
    """Build a transformers BitsAndBytesConfig for the requested CUDA quantization mode."""
    from transformers import BitsAndBytesConfig
    import torch

    if mode == "int4":
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
    if mode == "int8":
        return BitsAndBytesConfig(load_in_8bit=True)
    raise ValueError(f"bnb_config does not support mode={mode!r}; use 'int4' or 'int8'")


def convert_to_gguf(
    hf_model_dir: str,
    output_path: str,
    quant_type: str = "Q4_K_M",
    llama_cpp_repo: str = "llama.cpp",
) -> Path:
    """Convert a HuggingFace checkpoint to a quantized GGUF file for llama.cpp CPU inference.

    Requires a local clone of llama.cpp with its `convert_hf_to_gguf.py` script and
    `llama-quantize` binary built. This function shells out to that toolchain rather
    than reimplementing GGUF quantization, since the reference converter is
    maintained upstream and vision-model support in llama.cpp is a moving target.

    Raises FileNotFoundError if the llama.cpp toolchain isn't present at `llama_cpp_repo`.
    """
    repo = Path(llama_cpp_repo)
    convert_script = repo / "convert_hf_to_gguf.py"
    quantize_bin = repo / "llama-quantize"

    if not convert_script.exists():
        raise FileNotFoundError(
            f"{convert_script} not found. Clone https://github.com/ggerganov/llama.cpp "
            f"and build it to use the CPU GGUF quantization path."
        )
    if not quantize_bin.exists() and shutil.which("llama-quantize") is None:
        raise FileNotFoundError(
            f"llama-quantize binary not found at {quantize_bin} or on PATH. "
            f"Build llama.cpp with `make llama-quantize` first."
        )

    fp16_gguf = Path(output_path).with_suffix(".fp16.gguf")
    subprocess.run(
        ["python3", str(convert_script), hf_model_dir, "--outfile", str(fp16_gguf), "--outtype", "f16"],
        check=True,
    )

    quantize_cmd = str(quantize_bin) if quantize_bin.exists() else "llama-quantize"
    subprocess.run(
        [quantize_cmd, str(fp16_gguf), str(output_path), quant_type],
        check=True,
    )
    return Path(output_path)


def load_gguf_model(gguf_path: str, n_ctx: int = 4096, n_threads: int | None = None):
    """Load a GGUF-quantized model for CPU inference via llama-cpp-python."""
    from llama_cpp import Llama

    return Llama(model_path=gguf_path, n_ctx=n_ctx, n_threads=n_threads, logits_all=False)


def gguf_json_grammar(schema: dict | None = None):
    """Build a llama.cpp grammar object that constrains generation to valid JSON.

    This is the third constrained-decoding path described in `constrained.py`: on
    the GGUF/CPU backend, grammar enforcement happens inside llama.cpp's sampler in
    C++ over the full vocabulary, so it avoids both the Python per-step cost and the
    top-k approximation the built-in HuggingFace logits processor makes.

    Pass the result to `Llama.__call__(..., grammar=...)`.
    """
    from llama_cpp import LlamaGrammar

    from .constrained import json_schema_to_gbnf

    return LlamaGrammar.from_string(json_schema_to_gbnf(schema))
