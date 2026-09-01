"""Extraction accuracy as a function of quantization level — `vf accuracy`.

Latency benchmarks alone can't answer the question that decides a deployment:
*what does the speedup cost you?* This module runs the same labeled extraction set
through every available (backend, quantization, model-size) configuration and
reports three things per configuration:

  - **schema validity rate** — fraction of samples that produced parseable JSON
    containing the requested keys. This is the number that decides whether output
    can be piped into a downstream system at all.
  - **field accuracy (exact)** — fraction of individual fields matching ground
    truth after whitespace/case/dash normalization.
  - **field accuracy (fuzzy)** — same, scored with a character-similarity
    threshold. A 2B VLM reading a scanned document frequently drops one character
    from a long identifier ("SYN-0041729" for "SYN-00417293"). Exact match calls
    that a total failure and fuzzy match calls it a near-miss; reporting both
    prevents either metric from flattering the model.

Ground truth comes from `benchmarks/labeled_set.json`, whose labels are the exact
strings drawn into the synthetic images by `examples/generate_sample_images.py` —
correct by construction, no annotation error. The set is deliberately small; it is
sized to expose *relative* differences between quantization levels, not to
establish an absolute accuracy claim.

Each configuration is measured in a separate subprocess, for the same reason as
in `bench.py`: loading several multi-GB models into one process on a 16GB machine
means the later ones are measured under memory pressure the earlier ones did not face.
"""

from __future__ import annotations

import difflib
import json
import re
import subprocess
import sys
import unicodedata
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

FUZZY_THRESHOLD = 0.85
_MARKER = "__VF_ACC_JSON__"


@dataclass
class FieldOutcome:
    sample_id: str
    field: str
    expected: str
    got: Optional[str]
    exact: bool
    fuzzy: bool
    similarity: float


@dataclass
class AccuracyResult:
    label: str
    backend: str
    model_id: str
    constrained: bool
    quantization: Optional[str] = None
    skipped: bool = False
    reason: Optional[str] = None

    n_samples: int = 0
    n_schema_valid: int = 0
    n_fields: int = 0
    n_exact: int = 0
    n_fuzzy: int = 0
    n_repaired: int = 0
    mean_seconds: Optional[float] = None
    outcomes: list[dict] = field(default_factory=list)

    @property
    def schema_validity_rate(self) -> Optional[float]:
        return self.n_schema_valid / self.n_samples if self.n_samples else None

    @property
    def exact_accuracy(self) -> Optional[float]:
        return self.n_exact / self.n_fields if self.n_fields else None

    @property
    def fuzzy_accuracy(self) -> Optional[float]:
        return self.n_fuzzy / self.n_fields if self.n_fields else None

    def to_dict(self) -> dict:
        d = asdict(self)
        d["schema_validity_rate"] = self.schema_validity_rate
        d["exact_accuracy"] = self.exact_accuracy
        d["fuzzy_accuracy"] = self.fuzzy_accuracy
        return d


_DASHES = dict.fromkeys(map(ord, "‐‑‒–—―−"), "-")


def normalize(value) -> str:
    """Normalization applied before comparison.

    Deliberately conservative: it folds case, whitespace, and the several Unicode
    dash characters that a model may substitute for the em-dash rendered in the
    image. It does NOT strip punctuation or digits, because those carry the
    meaning in an MRN, a lot number, or an ICD-10 code.
    """
    if value is None:
        return ""
    if isinstance(value, (list, tuple)):
        value = " ".join(str(v) for v in value)
    text = unicodedata.normalize("NFKC", str(value)).translate(_DASHES)
    return re.sub(r"\s+", " ", text).strip().lower()


def score_field(expected, got) -> tuple[bool, bool, float]:
    e, g = normalize(expected), normalize(got)
    if not e:
        return (False, False, 0.0)
    if e == g:
        return (True, True, 1.0)
    similarity = difflib.SequenceMatcher(None, e, g).ratio()
    return (False, similarity >= FUZZY_THRESHOLD, similarity)


def load_labeled_set(path: Optional[str] = None, root: Optional[Path] = None) -> tuple[list[dict], Path]:
    root = root or Path(__file__).parent.parent
    path = Path(path) if path else root / "benchmarks" / "labeled_set.json"
    data = json.loads(Path(path).read_text())
    return data["samples"], root


def _run_worker(backend: str, model_id: str, label: str, constrained: bool,
                samples: list[dict], root: str, max_new_tokens: int,
                quant_mode: Optional[str] = None) -> AccuracyResult:
    """Evaluate one configuration in this process."""
    import time

    import torch

    from .engine import HardwareTier
    from .pipeline import VisionFlow

    result = AccuracyResult(label=label, backend=backend, model_id=model_id, constrained=constrained)

    if backend == "cuda" and not torch.cuda.is_available():
        result.skipped, result.reason = True, "no CUDA device on this machine"
        return result
    if backend == "mps" and not torch.backends.mps.is_available():
        result.skipped, result.reason = True, "MPS not available on this machine"
        return result

    tier = HardwareTier(backend)
    vf = VisionFlow(model_id=model_id, hardware=tier, force_cpu=(tier == HardwareTier.CPU))
    vf.engine.quant_mode = quant_mode
    try:
        load_stats = vf.load()
        result.quantization = load_stats.quantization
    except Exception as e:
        result.skipped, result.reason = True, f"model load failed: {type(e).__name__}: {e}"
        return result

    root_path = Path(root)
    durations = []

    for sample in samples:
        image_path = root_path / sample["image"]
        truth = sample["ground_truth"]
        start = time.perf_counter()
        try:
            extraction = vf.json(
                image_path, prompt=sample["prompt"], schema=sample["schema"],
                max_new_tokens=max_new_tokens, constrained=constrained,
            )
        except Exception as e:
            result.n_samples += 1
            result.n_fields += len(truth)
            result.outcomes.append({"sample_id": sample["id"], "error": f"{type(e).__name__}: {e}"})
            continue
        durations.append(time.perf_counter() - start)

        result.n_samples += 1
        if extraction.repaired:
            result.n_repaired += 1

        parsed = extraction.parsed if extraction.ok and isinstance(extraction.parsed, dict) else None
        # "Schema valid" means parseable AND carrying every requested key — a JSON
        # object with none of the asked-for fields is not usable downstream.
        if parsed is not None and all(k in parsed for k in sample["schema"]):
            result.n_schema_valid += 1

        for key, expected in truth.items():
            got = parsed.get(key) if parsed else None
            exact, fuzzy, similarity = score_field(expected, got)
            result.n_fields += 1
            result.n_exact += int(exact)
            result.n_fuzzy += int(fuzzy)
            result.outcomes.append(asdict(FieldOutcome(
                sample_id=sample["id"], field=key, expected=str(expected),
                got=None if got is None else str(got),
                exact=exact, fuzzy=fuzzy, similarity=round(similarity, 3),
            )))

    if durations:
        result.mean_seconds = sum(durations) / len(durations)
    return result


def available_configs(model_id: Optional[str] = None, include_constrained: bool = True) -> list[dict]:
    import torch

    from .engine import MODEL_TIERS, ModelSize, select_model

    base = model_id or select_model()
    configs = [
        {"backend": "cuda", "model_id": base, "label": "CUDA / INT4 (bnb nf4)", "quant_mode": "int4"},
        {"backend": "cuda", "model_id": base, "label": "CUDA / INT8 (bnb)", "quant_mode": "int8"},
        {"backend": "mps", "model_id": base, "label": "Apple MPS / fp16", "quant_mode": None},
        {"backend": "cpu", "model_id": base, "label": "CPU / fp32", "quant_mode": None},
    ]
    small = MODEL_TIERS[ModelSize.SMALL]
    if small != base and torch.backends.mps.is_available():
        configs.append({"backend": "mps", "model_id": small,
                        "label": "Apple MPS / fp16 (SmolVLM-256M)", "quant_mode": None})

    out = [dict(c, constrained=False) for c in configs]
    if include_constrained:
        # Every backend gets a constrained counterpart, CUDA included. An earlier
        # version skipped CUDA here on the reasoning that it would be skipped
        # anyway on the dev machine — which meant that on an actual GPU box, the
        # comparison that matters most would silently not run. Unavailable
        # configurations are already handled by the worker's skip path; there is
        # no need to second-guess availability when building the list.
        for c in configs:
            out.append(dict(c, constrained=True, label=c["label"] + " + constrained"))
    return out


def run_accuracy(
    labeled_set: Optional[str] = None,
    model_id: Optional[str] = None,
    max_new_tokens: int = 512,
    isolate: bool = True,
    include_constrained: bool = True,
    timeout: int = 5400,
) -> dict:
    samples, root = load_labeled_set(labeled_set)
    configs = available_configs(model_id, include_constrained=include_constrained)
    results = []

    for config in configs:
        print(f"[accuracy] {config['label']} ...", file=sys.stderr, flush=True)
        if not isolate:
            results.append(_run_worker(
                config["backend"], config["model_id"], config["label"],
                config["constrained"], samples, str(root), max_new_tokens,
                config.get("quant_mode"),
            ).to_dict())
            continue

        payload = json.dumps({
            **config, "samples": samples, "root": str(root), "max_new_tokens": max_new_tokens,
        })
        proc = subprocess.run(
            [sys.executable, "-m", "visionflow.accuracy", "--_worker", payload],
            capture_output=True, text=True, timeout=timeout,
        )
        parsed = None
        for line in proc.stdout.splitlines():
            if line.startswith(_MARKER):
                try:
                    parsed = json.loads(line[len(_MARKER):])
                except Exception:
                    parsed = None
                break
        if parsed is None:
            tail = (proc.stderr or "").strip().splitlines()[-3:]
            parsed = AccuracyResult(
                label=config["label"], backend=config["backend"], model_id=config["model_id"],
                constrained=config["constrained"], skipped=True,
                reason=f"worker exited {proc.returncode}: {' / '.join(tail) or 'no output'}",
            ).to_dict()
        results.append(parsed)

    from .bench import _host_info

    return {
        "methodology": {
            "labeled_set": str(labeled_set or "benchmarks/labeled_set.json"),
            "n_samples": len(samples),
            "n_fields": sum(len(s["ground_truth"]) for s in samples),
            "fuzzy_threshold": FUZZY_THRESHOLD,
            "max_new_tokens": max_new_tokens,
            "process_isolation": isolate,
        },
        "host": _host_info(),
        "results": results,
    }


def to_markdown(report: dict) -> str:
    meth = report.get("methodology", {})
    lines = [
        "| Configuration | Quant | Schema-valid | Field acc. (exact) | Field acc. (fuzzy) | Repairs used | Mean s/sample |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in report.get("results", []):
        if r.get("skipped"):
            lines.append(
                f"| {r['label']} | — | *not measured — {r.get('reason', 'unavailable')}* | — | — | — | — |"
            )
            continue
        pct = lambda v: f"{v * 100:.0f}%" if isinstance(v, (int, float)) else "—"
        mean_s = r.get("mean_seconds")
        lines.append(
            f"| {r['label']} | {r.get('quantization') or '—'} | "
            f"{pct(r.get('schema_validity_rate'))} ({r.get('n_schema_valid', 0)}/{r.get('n_samples', 0)}) | "
            f"{pct(r.get('exact_accuracy'))} ({r.get('n_exact', 0)}/{r.get('n_fields', 0)}) | "
            f"{pct(r.get('fuzzy_accuracy'))} ({r.get('n_fuzzy', 0)}/{r.get('n_fields', 0)}) | "
            f"{r.get('n_repaired', 0)} | "
            f"{f'{mean_s:.1f}' if isinstance(mean_s, (int, float)) else '—'} |"
        )
    lines.append(
        f"\n{meth.get('n_samples', '?')} images / {meth.get('n_fields', '?')} labeled fields. "
        f"Exact = normalized string equality; fuzzy = character similarity ≥ "
        f"{meth.get('fuzzy_threshold', FUZZY_THRESHOLD)}. "
        "\"Schema-valid\" requires parseable JSON containing every requested key. "
        "Each configuration measured in an isolated subprocess.\n"
    )
    return "\n".join(lines)


def main(argv: Optional[list[str]] = None):
    import argparse

    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "--_worker":
        data = json.loads(argv[1])
        result = _run_worker(
            data["backend"], data["model_id"], data["label"], data["constrained"],
            data["samples"], data["root"], data["max_new_tokens"], data.get("quant_mode"),
        )
        print(_MARKER + json.dumps(result.to_dict()), flush=True)
        return

    parser = argparse.ArgumentParser(
        prog="vf accuracy",
        description="Measure extraction accuracy per quantization level against a labeled set",
    )
    parser.add_argument("--labeled-set", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--no-constrained", action="store_true",
                        help="Skip the grammar-constrained decoding comparison rows")
    parser.add_argument("--no-isolate", action="store_true")
    parser.add_argument("--out", default="benchmarks/results/accuracy_report.json")
    parser.add_argument("--markdown", default="benchmarks/results/accuracy_table.md")
    args = parser.parse_args(argv)

    report = run_accuracy(
        labeled_set=args.labeled_set, model_id=args.model,
        max_new_tokens=args.max_new_tokens, isolate=not args.no_isolate,
        include_constrained=not args.no_constrained,
    )
    table = to_markdown(report)
    print("\n" + table)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2))
    Path(args.markdown).parent.mkdir(parents=True, exist_ok=True)
    Path(args.markdown).write_text(table)
    print(f"[accuracy] wrote {args.out} and {args.markdown}", file=sys.stderr)


if __name__ == "__main__":
    main()
