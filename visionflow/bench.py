"""VisionFlow benchmark harness — `vf bench`.

Benchmarking methodology, stated up front so the numbers can be judged:

1. **Process isolation per configuration.** Each (backend, quantization, model)
   combination is measured in a *fresh subprocess*. This is not incidental: an
   earlier version of this repo measured MPS and then CPU in one process and
   reported ~15.3GB peak for both, because the CPU measurement inherited the MPS
   driver's still-live allocations. Peak-memory numbers are only meaningful when
   nothing else has already allocated on the device.

2. **Warmup runs are excluded.** The first generation on a backend pays lazy
   kernel compilation, cache population, and (on MPS) shader compilation. Those
   costs are real but they are *startup* costs, reported separately as model load
   time, not folded into steady-state latency.

3. **Tokens/sec alongside latency.** SmolVLM emits very different output lengths
   per image category — a chart description runs several times longer than a
   key-value form. Seconds-per-sample therefore conflates "this backend is slow"
   with "this image produced more tokens". Both are reported.

4. **p50 and p95, not just mean ± std.** Latency distributions here are strongly
   right-skewed, so a mean ± std implies a symmetric spread that does not exist.

5. **Unavailable configurations are recorded, not omitted.** A machine with no
   CUDA device emits a `skipped` row with a reason. Silence would read as
   "not applicable"; an explicit skip reads as "not measured here".

Caveat on the memory columns: `peak_rss_mb` is sampled process RSS, and safetensors
weights are memory-mapped, so pages the kernel has not faulted in (or has evicted)
do not count toward RSS. RSS is therefore a *lower* bound on a model's true memory
demand — it answers "how much did this process actually keep resident", not "how
much does this checkpoint weigh". `device_peak_mb` comes from the backend's own
allocator and does not have this problem, but only exists for MPS and CUDA.
"""

from __future__ import annotations

import json
import os
import statistics
import subprocess
import sys
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

DEFAULT_PROMPT = "Describe the key information in this document or image."
WARMUP_RUNS = 1


@dataclass
class BenchConfig:
    """One measurable configuration."""

    backend: str            # "cuda" | "mps" | "cpu"
    model_id: str
    label: str
    quant_hint: Optional[str] = None   # "int4" | "int8" | "gguf-q4" | None (native)


@dataclass
class BenchResult:
    label: str
    backend: str
    model_id: str
    quantization: Optional[str] = None
    skipped: bool = False
    reason: Optional[str] = None

    load_seconds: Optional[float] = None
    n_measured: int = 0
    latencies: list[float] = field(default_factory=list)
    tokens: list[int] = field(default_factory=list)

    mean_s: Optional[float] = None
    p50_s: Optional[float] = None
    p95_s: Optional[float] = None
    std_s: Optional[float] = None
    tokens_per_second: Optional[float] = None
    peak_rss_mb: Optional[float] = None
    device_peak_mb: Optional[float] = None
    device_memory_note: Optional[str] = None

    def finalize(self):
        if not self.latencies:
            return
        ordered = sorted(self.latencies)
        self.n_measured = len(ordered)
        self.mean_s = statistics.mean(ordered)
        self.p50_s = _percentile(ordered, 50)
        self.p95_s = _percentile(ordered, 95)
        self.std_s = statistics.stdev(ordered) if len(ordered) > 1 else 0.0
        total_tokens, total_time = sum(self.tokens), sum(self.latencies)
        self.tokens_per_second = (total_tokens / total_time) if total_time > 0 else None


def _percentile(ordered: list[float], pct: float) -> float:
    """Linear-interpolation percentile. Small n here (typically 9-27), so the
    interpolation choice matters; documenting it beats an unexplained number."""
    if len(ordered) == 1:
        return ordered[0]
    rank = (pct / 100) * (len(ordered) - 1)
    low, high = int(rank), min(int(rank) + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (rank - low)


class _RSSSampler:
    """Samples process RSS on a background thread to capture true peak.

    Reading RSS only at the end misses the peak, which for VLM inference occurs
    during the forward pass over image patches, not after generation returns.
    """

    def __init__(self, interval: float = 0.05):
        self.interval = interval
        self.peak_mb = 0.0
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        try:
            import psutil

            self._proc = psutil.Process(os.getpid())
        except Exception:
            self._proc = None

    def _run(self):
        while not self._stop.is_set():
            try:
                rss = self._proc.memory_info().rss / (1024**2)
                self.peak_mb = max(self.peak_mb, rss)
            except Exception:
                return
            self._stop.wait(self.interval)

    def __enter__(self):
        if self._proc is not None:
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        return False


def available_configs(model_id: Optional[str] = None) -> list[BenchConfig]:
    """Enumerate every configuration worth attempting on this machine.

    Configurations that cannot run here are still returned — the worker records
    them as skipped with a reason, so the output table shows the gap explicitly.
    """
    import torch

    from .engine import MODEL_TIERS, ModelSize, select_model

    base = model_id or select_model()
    configs = [
        BenchConfig("cuda", base, "CUDA / INT4 (bnb nf4)", quant_hint="int4"),
        BenchConfig("cuda", base, "CUDA / INT8 (bnb)", quant_hint="int8"),
        BenchConfig("mps", base, "Apple MPS / fp16"),
        BenchConfig("cpu", base, "CPU / fp32"),
    ]
    # The small tier is what actually runs on a Pi-class device; measuring it on
    # the dev machine's CPU gives an honest lower bound for that deployment story.
    small = MODEL_TIERS[ModelSize.SMALL]
    if small != base:
        configs.append(BenchConfig("cpu", small, "CPU / fp32 (SmolVLM-256M)"))
        if torch.backends.mps.is_available():
            configs.append(BenchConfig("mps", small, "Apple MPS / fp16 (SmolVLM-256M)"))
    return configs


def _run_worker(config: BenchConfig, images: list[str], prompt: str, repeats: int,
                max_new_tokens: int) -> BenchResult:
    """Measure one configuration in *this* process. Called inside the subprocess."""
    import torch

    from .engine import HardwareTier, ModelEngine

    result = BenchResult(label=config.label, backend=config.backend, model_id=config.model_id)

    if config.backend == "cuda" and not torch.cuda.is_available():
        result.skipped, result.reason = True, "no CUDA device on this machine"
        return result
    if config.backend == "mps" and not torch.backends.mps.is_available():
        result.skipped, result.reason = True, "MPS not available on this machine"
        return result

    from PIL import Image

    tier = HardwareTier(config.backend)
    engine = ModelEngine(model_id=config.model_id, hardware=tier,
                         force_cpu=(tier == HardwareTier.CPU),
                         quant_mode=config.quant_hint)

    try:
        load_stats = engine.load()
    except Exception as e:
        result.skipped, result.reason = True, f"model load failed: {type(e).__name__}: {e}"
        return result

    result.load_seconds = load_stats.load_seconds
    result.quantization = load_stats.quantization

    loaded = [Image.open(p).convert("RGB") for p in images]

    with _RSSSampler() as sampler:
        try:
            for _ in range(WARMUP_RUNS):
                engine.generate_with_stats(loaded[0], prompt, max_new_tokens=max_new_tokens)
            for _ in range(repeats):
                for img in loaded:
                    stats = engine.generate_with_stats(img, prompt, max_new_tokens=max_new_tokens)
                    result.latencies.append(stats.seconds)
                    result.tokens.append(stats.new_tokens)
        except Exception as e:
            result.skipped, result.reason = True, f"generation failed: {type(e).__name__}: {e}"
            return result

    result.peak_rss_mb = sampler.peak_mb or None
    result.device_peak_mb, result.device_memory_note = _device_peak_memory(tier)
    result.finalize()
    return result


def _device_peak_memory(tier) -> tuple[Optional[float], Optional[str]]:
    import torch

    from .engine import HardwareTier

    try:
        if tier == HardwareTier.CUDA:
            return torch.cuda.max_memory_allocated() / (1024**2), "torch.cuda.max_memory_allocated"
        if tier == HardwareTier.MPS:
            return torch.mps.driver_allocated_memory() / (1024**2), "torch.mps.driver_allocated_memory"
    except Exception:
        pass
    return None, None


def run_benchmark(
    images: list[str],
    prompt: str = DEFAULT_PROMPT,
    repeats: int = 3,
    max_new_tokens: int = 256,
    model_id: Optional[str] = None,
    configs: Optional[list[BenchConfig]] = None,
    isolate: bool = True,
    timeout: int = 3600,
) -> dict:
    """Run every configuration and return a report dict.

    With `isolate=True` (the default and the only setting whose memory numbers
    should be trusted), each configuration runs in a fresh subprocess.
    """
    configs = configs or available_configs(model_id)
    results: list[BenchResult] = []

    for config in configs:
        print(f"[bench] {config.label} ...", file=sys.stderr, flush=True)
        if not isolate:
            results.append(_run_worker(config, images, prompt, repeats, max_new_tokens))
            continue

        payload = json.dumps({
            "config": asdict(config),
            "images": images,
            "prompt": prompt,
            "repeats": repeats,
            "max_new_tokens": max_new_tokens,
        })
        proc = subprocess.run(
            [sys.executable, "-m", "visionflow.bench", "--_worker", payload],
            capture_output=True, text=True, timeout=timeout,
        )
        parsed = _parse_worker_output(proc.stdout)
        if parsed is None:
            tail = (proc.stderr or "").strip().splitlines()[-3:]
            results.append(BenchResult(
                label=config.label, backend=config.backend, model_id=config.model_id,
                skipped=True,
                reason=f"worker exited {proc.returncode}: {' / '.join(tail) or 'no output'}",
            ))
        else:
            results.append(parsed)

    return {
        "methodology": {
            "process_isolation": isolate,
            "warmup_runs_excluded": WARMUP_RUNS,
            "repeats_per_image": repeats,
            "n_images": len(images),
            "max_new_tokens": max_new_tokens,
            "prompt": prompt,
            "images": images,
        },
        "host": _host_info(),
        "results": [asdict(r) for r in results],
    }


_MARKER = "__VF_BENCH_JSON__"


def _parse_worker_output(stdout: str) -> Optional[BenchResult]:
    """Pull the result payload out of worker stdout.

    Marker-delimited rather than "parse the whole of stdout": transformers and
    tokenizers write progress bars and warnings to stdout that would otherwise
    corrupt the JSON.
    """
    for line in stdout.splitlines():
        if line.startswith(_MARKER):
            try:
                return BenchResult(**json.loads(line[len(_MARKER):]))
            except Exception:
                return None
    return None


def _host_info() -> dict:
    import platform

    import torch

    from .engine import total_memory_gb

    info = {
        "platform": platform.platform(),
        "processor": platform.processor() or platform.machine(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "total_ram_gb": round(total_memory_gb() or 0, 1) or None,
        "cuda_available": torch.cuda.is_available(),
        "mps_available": bool(torch.backends.mps.is_available()),
    }
    if torch.cuda.is_available():
        try:
            info["cuda_device"] = torch.cuda.get_device_name(0)
        except Exception:
            pass
    return info


def to_markdown(report: dict) -> str:
    """Render the report as the README table."""
    host = report.get("host", {})
    meth = report.get("methodology", {})
    lines = [
        "| Configuration | Model | Quant | Load (s) | p50 (s) | p95 (s) | mean ± std (s) | tok/s | Peak RSS (MB) | Device mem (MB) |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in report.get("results", []):
        model_short = r["model_id"].split("/")[-1]
        if r.get("skipped"):
            lines.append(
                f"| {r['label']} | {model_short} | — | — | — | — | "
                f"*not measured — {r.get('reason', 'unavailable')}* | — | — | — |"
            )
            continue
        f2 = lambda v: f"{v:.2f}" if isinstance(v, (int, float)) else "—"
        f0 = lambda v: f"{v:,.0f}" if isinstance(v, (int, float)) else "—"
        lines.append(
            f"| {r['label']} | {model_short} | {r.get('quantization') or '—'} | "
            f"{f2(r.get('load_seconds'))} | {f2(r.get('p50_s'))} | {f2(r.get('p95_s'))} | "
            f"{f2(r.get('mean_s'))} ± {f2(r.get('std_s'))} | "
            f"{f2(r.get('tokens_per_second'))} | {f0(r.get('peak_rss_mb'))} | "
            f"{f0(r.get('device_peak_mb'))} |"
        )

    footer = (
        f"\nHost: {host.get('platform', 'unknown')} · {host.get('processor', '')} · "
        f"{host.get('total_ram_gb', '?')}GB RAM · torch {host.get('torch', '?')}. "
        f"{meth.get('n_images', '?')} images × {meth.get('repeats_per_image', '?')} repeats, "
        f"{meth.get('warmup_runs_excluded', 0)} warmup run(s) excluded, "
        f"max_new_tokens={meth.get('max_new_tokens', '?')}, "
        f"{'each configuration in an isolated subprocess' if meth.get('process_isolation') else 'all configurations in one process (memory figures not independent)'}.\n"
    )
    return "\n".join(lines) + "\n" + footer


def _worker_main(payload: str):
    data = json.loads(payload)
    config = BenchConfig(**data["config"])
    result = _run_worker(
        config, data["images"], data["prompt"], data["repeats"], data["max_new_tokens"]
    )
    print(_MARKER + json.dumps(asdict(result)), flush=True)


def main(argv: Optional[list[str]] = None):
    import argparse

    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "--_worker":
        _worker_main(argv[1])
        return

    parser = argparse.ArgumentParser(
        prog="vf bench", description="Benchmark VisionFlow across backends and quantization levels"
    )
    parser.add_argument("--images", nargs="*", default=None,
                        help="Images to benchmark (default: the bundled sample set)")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--model", default=None, help="Override the HF model id")
    parser.add_argument("--no-isolate", action="store_true",
                        help="Run all configs in one process (faster, but memory numbers are not independent)")
    parser.add_argument("--out", default="benchmarks/results/bench_report.json")
    parser.add_argument("--markdown", default="benchmarks/results/bench_table.md")
    args = parser.parse_args(argv)

    images = args.images or _default_images()
    if not images:
        parser.error("no images found — pass --images, or run examples/generate_sample_images.py")

    report = run_benchmark(
        images=images, prompt=args.prompt, repeats=args.repeats,
        max_new_tokens=args.max_new_tokens, model_id=args.model,
        isolate=not args.no_isolate,
    )

    table = to_markdown(report)
    print("\n" + table)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2))
    Path(args.markdown).parent.mkdir(parents=True, exist_ok=True)
    Path(args.markdown).write_text(table)
    print(f"[bench] wrote {args.out} and {args.markdown}", file=sys.stderr)


def _default_images() -> list[str]:
    sample_dir = Path(__file__).parent.parent / "examples" / "sample_images"
    return sorted(str(p) for p in sample_dir.glob("*.png"))


if __name__ == "__main__":
    main()
