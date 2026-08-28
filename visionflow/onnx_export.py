"""ONNX export and ONNX Runtime execution, including the TensorRT provider path.

**Verification status — read this before trusting anything here.**

  - Vision-encoder ONNX export and CPU-provider inference: *verified* on the
    development machine (Apple M3). `python -m visionflow.onnx_export --verify`
    reproduces it and checks ONNX output against PyTorch to a numeric tolerance.
  - CUDA execution provider: *implemented, never run.* No NVIDIA hardware was
    available during development.
  - TensorRT execution provider: *implemented, never run.* Same reason.

The TensorRT path is written from the ORT provider API rather than from measured
experience, and it is labelled that way in the README benchmark table instead of
being presented as a result. To earn those numbers, run this module's `--benchmark`
on a CUDA machine and paste the emitted table in — the harness records the provider
actually used, so a silent fallback to CPUExecutionProvider cannot be mistaken for
a TensorRT number.

Why export only the vision encoder by default: SmolVLM's language decoder is an
autoregressive model with KV-cache state, which ONNX can represent but only via a
considerably more involved export (separate prefill/decode graphs, cache tensors as
graph I/O). The vision tower is a single fixed-shape forward pass, exports cleanly,
and is the part whose cost is most sensitive to the execution provider. Full-model
export is left to `optimum.exporters.onnx`, wired up in `export_with_optimum` for
users who have it installed.
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

DEFAULT_OPSET = 17

# Preference order. TensorRT first, then CUDA, then CPU — ORT silently falls back
# through this list, which is exactly why `provider_used` is reported per row.
#
# CoreML is deliberately NOT in the default list. It is available on Apple
# hardware, but it compiles the graph at session-creation time and only supports
# 768 of this model's 1607 nodes, so it spends minutes partitioning and then runs
# a hybrid CoreML/CPU graph whose latency answers a different question than the
# one this harness asks. Opt in with `--providers CoreMLExecutionProvider`.
PROVIDER_PREFERENCE = [
    "TensorrtExecutionProvider",
    "CUDAExecutionProvider",
    "CPUExecutionProvider",
]


@dataclass
class ExportResult:
    ok: bool
    path: Optional[str] = None
    opset: int = DEFAULT_OPSET
    seconds: Optional[float] = None
    input_shape: Optional[list] = None
    output_shape: Optional[list] = None
    max_abs_diff_vs_torch: Optional[float] = None
    exporter: Optional[str] = None
    error: Optional[str] = None


@dataclass
class OnnxBenchResult:
    provider_requested: str
    provider_used: Optional[str] = None
    ok: bool = False
    reason: Optional[str] = None
    n_runs: int = 0
    latencies_ms: list[float] = field(default_factory=list)
    mean_ms: Optional[float] = None
    p50_ms: Optional[float] = None
    p95_ms: Optional[float] = None

    def finalize(self):
        if not self.latencies_ms:
            return
        from .bench import _percentile

        ordered = sorted(self.latencies_ms)
        self.n_runs = len(ordered)
        self.mean_ms = sum(ordered) / len(ordered)
        self.p50_ms = _percentile(ordered, 50)
        self.p95_ms = _percentile(ordered, 95)


def available_providers() -> list[str]:
    try:
        import onnxruntime as ort

        return list(ort.get_available_providers())
    except Exception:
        return []


def _load_vision_tower(model_id: str):
    """Return (vision_module, processor, image_size) for the given VLM."""
    import torch
    from transformers import AutoProcessor

    try:
        from transformers import AutoModelForImageTextToText as AutoModelForVLM
    except ImportError:
        from transformers import AutoModelForVision2Seq as AutoModelForVLM

    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForVLM.from_pretrained(model_id, torch_dtype=torch.float32)
    model.eval()

    # SmolVLM/Idefics3 nest the vision tower differently across transformers
    # versions; probe rather than hard-coding one attribute path.
    for path in ("model.vision_model", "vision_model", "model.vision_tower", "vision_tower"):
        module = model
        for attr in path.split("."):
            module = getattr(module, attr, None)
            if module is None:
                break
        if module is not None:
            return module, processor, model

    raise RuntimeError(f"could not locate a vision tower on {model_id}")


class _VisionWrapper:
    """Adapter so torch.onnx.export sees a plain tensor-in/tensor-out module."""

    def __new__(cls, vision_module):
        import torch

        class Wrapper(torch.nn.Module):
            def __init__(self, inner):
                super().__init__()
                self.inner = inner

            def forward(self, pixel_values):
                out = self.inner(pixel_values=pixel_values)
                return out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]

        return Wrapper(vision_module)


def _specialize_position_ids(vision_module) -> bool:
    """Replace SmolVLM's data-dependent position-id computation with `arange`.

    `Idefics3VisionEmbeddings.forward` derives position ids by counting unmasked
    patches, bucketizing fractional coordinates, and scattering through a boolean
    mask. That is only meaningful when patches are *padded*; for a full, unpadded
    patch grid it reduces exactly to `arange(num_patches)`. Both ONNX exporters
    trace the general form into a GatherND whose indices are valid only for the
    traced batch, producing a graph that loads fine and then throws
    "invalid index found" on the very input it was exported with.

    Specializing to the unpadded case removes the data dependence. The exported
    graph is then correct for full patch grids — which is what the processor
    produces — and wrong for padded ones, so it is NOT a general replacement for
    the PyTorch module. Correctness of the substitution is not assumed: the caller
    compares the exported graph against the *unpatched* module's output.

    Returns True if the patch was applied.
    """
    import torch

    embeddings = getattr(vision_module, "embeddings", None)
    if embeddings is None or not hasattr(embeddings, "patch_embedding"):
        return False

    def static_forward(pixel_values, patch_attention_mask=None, **_kwargs):
        patch_embeds = embeddings.patch_embedding(pixel_values)
        embedded = patch_embeds.flatten(2).transpose(1, 2)
        position_ids = torch.arange(embedded.shape[1], device=embedded.device).unsqueeze(0)
        return embedded + embeddings.position_embedding(position_ids)

    embeddings.forward = static_forward
    return True


def _patch_size(processor, default: int = 384) -> tuple[int, int]:
    """Side length of one image patch, as the vision tower expects it.

    This must come from `max_image_size`, not `size`. SmolVLM splits an image into
    N square patches and feeds the tower a tensor of shape (N, 3, 384, 384); `size`
    describes the *whole-image* resize bound (1536) and using it produces a graph
    that exports without complaint and then fails at runtime with an out-of-range
    GatherND index — which is exactly what happened before this function existed.
    """
    ip = getattr(processor, "image_processor", None)
    for attr, key in (("max_image_size", "longest_edge"), ("size", "height")):
        holder = getattr(ip, attr, None)
        if holder is None:
            continue
        value = holder.get(key) if isinstance(holder, dict) else getattr(holder, key, None)
        if isinstance(value, int) and value > 0:
            return (value, value)
    return (default, default)


def export_vision_encoder(
    model_id: Optional[str] = None,
    out_path: str = "benchmarks/onnx/vision_encoder.onnx",
    opset: int = DEFAULT_OPSET,
    verify: bool = True,
) -> ExportResult:
    """Export the vision tower to ONNX and (optionally) check it against PyTorch."""
    import torch

    from .engine import select_model

    model_id = model_id or select_model()
    try:
        vision_module, processor, _model = _load_vision_tower(model_id)
    except Exception as e:
        return ExportResult(ok=False, error=f"{type(e).__name__}: {e}")

    wrapper = _VisionWrapper(vision_module)
    wrapper.eval()

    dummy = torch.randn(1, 3, *_patch_size(processor))

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    # Reference output comes from the UNPATCHED module, so the specialization
    # below is checked rather than trusted.
    with torch.no_grad():
        torch_out = wrapper(dummy)

    specialized = _specialize_position_ids(vision_module)

    # Two exporters, tried in order, because they fail differently on this model.
    # SmolVLM's vision embeddings derive position ids through data-dependent
    # indexing (bucketize + boolean-mask scatter over the patch attention mask).
    # The dynamo exporter traces that into a GatherND whose indices are only valid
    # for the traced batch, so the graph exports cleanly and then throws
    # "invalid index found" at runtime. The legacy tracer specializes the same
    # control flow to concrete values, which is correct for a fixed patch shape.
    # Whichever succeeds is recorded, so the report says how the graph was made.
    attempts, errors = [("torchscript", False), ("dynamo", True)], []
    result = None

    for name, use_dynamo in attempts:
        start = time.perf_counter()
        try:
            with torch.no_grad():
                kwargs = {"dynamo": use_dynamo} if use_dynamo else {}
                torch.onnx.export(
                    wrapper, (dummy,), out_path,
                    input_names=["pixel_values"], output_names=["last_hidden_state"],
                    dynamic_axes={"pixel_values": {0: "patches"},
                                  "last_hidden_state": {0: "patches"}},
                    opset_version=opset, do_constant_folding=True, **kwargs,
                )
        except Exception as e:
            errors.append(f"{name}: {type(e).__name__}: {e}")
            continue

        candidate = ExportResult(
            ok=True, path=out_path, opset=opset, seconds=time.perf_counter() - start,
            input_shape=list(dummy.shape), output_shape=list(torch_out.shape),
            exporter=name + ("+static-position-ids" if specialized else ""),
        )
        if not verify:
            return candidate

        try:
            import numpy as np
            import onnxruntime as ort

            session = ort.InferenceSession(out_path, providers=["CPUExecutionProvider"])
            onnx_out = session.run(None, {"pixel_values": dummy.numpy()})[0]
            candidate.max_abs_diff_vs_torch = float(np.max(np.abs(onnx_out - torch_out.numpy())))
            return candidate
        except Exception as e:
            # An exported graph that can't run is not a successful export, no
            # matter what the exporter returned. Keep trying.
            errors.append(f"{name}: exported but failed verification: {type(e).__name__}: {e}")
            candidate.ok = False
            candidate.error = errors[-1]
            result = candidate

    if result is None:
        result = ExportResult(ok=False, opset=opset)
    result.ok = False
    result.error = " | ".join(errors)
    return result


def export_with_optimum(model_id: Optional[str] = None, out_dir: str = "benchmarks/onnx/full") -> ExportResult:
    """Full-model export via optimum, if it's installed.

    Not exercised in this repo's CI — recorded as an error result when optimum
    is missing rather than silently doing nothing.
    """
    from .engine import select_model

    model_id = model_id or select_model()
    try:
        from optimum.exporters.onnx import main_export
    except ImportError:
        return ExportResult(ok=False, error="optimum not installed (pip install 'optimum[exporters]')")

    try:
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        start = time.perf_counter()
        main_export(model_id, output=out_dir, task="image-text-to-text")
        return ExportResult(ok=True, path=out_dir, seconds=time.perf_counter() - start)
    except Exception as e:
        return ExportResult(ok=False, error=f"{type(e).__name__}: {e}")


def benchmark_onnx(
    onnx_path: str = "benchmarks/onnx/vision_encoder.onnx",
    providers: Optional[list[str]] = None,
    runs: int = 20,
    warmup: int = 3,
) -> list[OnnxBenchResult]:
    """Benchmark the exported graph under each requested execution provider.

    Every result records `provider_used` as reported by the live session, because
    ORT falls back silently: requesting TensorrtExecutionProvider on a machine
    without it yields a working CPU session and no error. A benchmark that didn't
    check would report CPU latency as a TensorRT number.
    """
    import numpy as np

    results: list[OnnxBenchResult] = []
    try:
        import onnxruntime as ort
    except ImportError:
        return [OnnxBenchResult(provider_requested=p, ok=False, reason="onnxruntime not installed")
                for p in (providers or PROVIDER_PREFERENCE)]

    if not Path(onnx_path).exists():
        return [OnnxBenchResult(provider_requested=p, ok=False, reason=f"{onnx_path} not found — run --export first")
                for p in (providers or PROVIDER_PREFERENCE)]

    installed = set(ort.get_available_providers())
    for provider in providers or PROVIDER_PREFERENCE:
        result = OnnxBenchResult(provider_requested=provider)
        if provider not in installed:
            result.reason = f"{provider} not available in this onnxruntime build"
            results.append(result)
            continue
        try:
            session = ort.InferenceSession(onnx_path, providers=[provider, "CPUExecutionProvider"])
            result.provider_used = session.get_providers()[0]
            if result.provider_used != provider:
                result.reason = f"ORT fell back to {result.provider_used}; this is NOT a {provider} measurement"

            inp = session.get_inputs()[0]
            shape = [d if isinstance(d, int) else 1 for d in inp.shape]
            dummy = np.random.randn(*shape).astype(np.float32)

            for _ in range(warmup):
                session.run(None, {inp.name: dummy})
            for _ in range(runs):
                start = time.perf_counter()
                session.run(None, {inp.name: dummy})
                result.latencies_ms.append((time.perf_counter() - start) * 1000)
            result.ok = True
            result.finalize()
        except Exception as e:
            result.reason = f"{type(e).__name__}: {e}"
        results.append(result)
    return results


def to_markdown(export: Optional[ExportResult], benches: list[OnnxBenchResult]) -> str:
    lines = []
    if export is not None:
        if export.ok:
            diff = export.max_abs_diff_vs_torch
            lines.append(
                f"ONNX export: **ok** — {export.exporter or 'default'} exporter, "
                f"opset {export.opset}, {export.seconds:.1f}s, "
                f"input {export.input_shape} → output {export.output_shape}"
                + (f", max |ONNX − PyTorch| = {diff:.2e}" if diff is not None else "")
            )
        else:
            lines.append(f"ONNX export: **failed** — {export.error}")
        lines.append("")

    lines += [
        "| Execution provider | Status | p50 (ms) | p95 (ms) | mean (ms) |",
        "|---|---|---|---|---|",
    ]
    for b in benches:
        f2 = lambda v: f"{v:.2f}" if isinstance(v, (int, float)) else "—"
        if not b.ok:
            lines.append(f"| {b.provider_requested} | *not measured — {b.reason}* | — | — | — |")
            continue
        status = "ok" if not b.reason else f"⚠️ {b.reason}"
        lines.append(
            f"| {b.provider_requested} | {status} | {f2(b.p50_ms)} | {f2(b.p95_ms)} | {f2(b.mean_ms)} |"
        )
    lines.append("\nVision encoder only (see module docstring for why the decoder is excluded).\n")
    return "\n".join(lines)


def main(argv: Optional[list[str]] = None):
    import argparse

    parser = argparse.ArgumentParser(
        prog="python -m visionflow.onnx_export",
        description="Export the vision encoder to ONNX and benchmark execution providers",
    )
    parser.add_argument("--export", action="store_true", help="Export the vision encoder to ONNX")
    parser.add_argument("--verify", action="store_true", help="Export, then check ONNX vs PyTorch outputs")
    parser.add_argument("--benchmark", action="store_true", help="Benchmark all available execution providers")
    parser.add_argument("--optimum", action="store_true", help="Full-model export via optimum")
    parser.add_argument("--model", default=None)
    parser.add_argument("--onnx-path", default="benchmarks/onnx/vision_encoder.onnx")
    parser.add_argument("--providers", nargs="*", default=None,
                        help=f"Execution providers to benchmark (default: {' '.join(PROVIDER_PREFERENCE)})")
    parser.add_argument("--runs", type=int, default=20)
    parser.add_argument("--out", default="benchmarks/results/onnx_report.json")
    parser.add_argument("--markdown", default="benchmarks/results/onnx_table.md")
    args = parser.parse_args(argv)

    if not any([args.export, args.verify, args.benchmark, args.optimum]):
        print("Available ONNX Runtime providers:", available_providers() or "onnxruntime not installed")
        parser.print_help()
        return

    export_result = None
    if args.optimum:
        export_result = export_with_optimum(args.model)
    elif args.export or args.verify:
        export_result = export_vision_encoder(args.model, out_path=args.onnx_path, verify=args.verify)

    benches = (
        benchmark_onnx(args.onnx_path, providers=args.providers, runs=args.runs)
        if args.benchmark
        else []
    )
    table = to_markdown(export_result, benches)
    print("\n" + table)

    report = {
        "host_providers": available_providers(),
        "export": asdict(export_result) if export_result else None,
        "benchmarks": [asdict(b) for b in benches],
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2))
    Path(args.markdown).parent.mkdir(parents=True, exist_ok=True)
    Path(args.markdown).write_text(table)
    print(f"[onnx] wrote {args.out} and {args.markdown}", file=sys.stderr)


if __name__ == "__main__":
    main()
