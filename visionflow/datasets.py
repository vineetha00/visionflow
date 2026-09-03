"""Build labeled evaluation sets from public VQA datasets — `vf dataset`.

Why this exists: the hand-built set in `benchmarks/labeled_set.json` has 21
fields. At that size one field is 4.8 percentage points, so almost no comparison
between configurations can reach statistical significance no matter how careful
the measurement is — the harness ended up far more precise than its sample size
justified. Scaling the labeled set is what turns the project's directional
claims into measurements.

Two sources, both public and ungated:

  docvqa  — nielsr/docvqa_1200_examples, scanned business documents (200 test
            examples). Closest to the medical/enterprise document framing.
  chartqa — HuggingFaceM4/ChartQA, chart and plot images (2,500 test examples).
            Numeric reading rather than text transcription.

Both annotate each image with a question and *several* acceptable answers, which
is why they are emitted with `{"any_of": [...]}` ground truth: the same address
is correct whether written "1128 SIXTEENTH ST., N. W." or lowercased, and
scoring against one arbitrary variant would charge the model for a transcription
convention rather than an error.

Structural caveat, stated because it changes what the numbers mean: these are
one-question-one-answer VQA items, so each sample contributes a single field
under a `{"answer": "string"}` schema. That measures extraction accuracy well
and makes n large, but it exercises the *multi-field* schema path far less than
`labeled_set.json` does — no field ordering, no partial-object failures, and
schema validity collapses to "did one key come back". Run both: the hand-built
set for schema behaviour, these for statistical power.

Images are materialized to disk as PNGs next to the emitted JSON, so a run is
reproducible without re-downloading and the accuracy harness needs no changes.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional

SOURCES = {
    "docvqa": {
        "hf_id": "nielsr/docvqa_1200_examples",
        "split": "test",
        "category": "scanned document",
        "description": "DocVQA — scanned business documents",
    },
    "chartqa": {
        "hf_id": "HuggingFaceM4/ChartQA",
        "split": "test",
        "category": "chart",
        "description": "ChartQA — charts and plots",
    },
}

PROMPT_SUFFIX = (
    " Answer with the exact value as it appears; do not explain or add units "
    "that are not shown."
)


def _question(row) -> Optional[str]:
    """DocVQA carries a multilingual dict of questions; ChartQA a bare string."""
    q = row.get("query")
    if isinstance(q, dict):
        return q.get("en") or next(iter(q.values()), None)
    return q if isinstance(q, str) else None


def _answers(row) -> list[str]:
    for key in ("answers", "label"):
        value = row.get(key)
        if isinstance(value, list) and value:
            return [str(v) for v in value]
        if isinstance(value, str) and value:
            return [value]
    return []


def build(
    source: str,
    n: int = 200,
    out_dir: Optional[str] = None,
    seed: int = 0,
) -> Path:
    """Materialize `n` samples from `source` into a labeled set. Returns its path."""
    if source not in SOURCES:
        raise ValueError(f"unknown source {source!r}; choose from {sorted(SOURCES)}")
    spec = SOURCES[source]

    from datasets import load_dataset

    out_dir = Path(out_dir or f"benchmarks/datasets/{source}_{n}")
    images_dir = out_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    # Shuffle before slicing. Taking the first n of a split risks inheriting
    # whatever ordering the dataset was built with (by document, by source
    # collection), which would make a "random subset" quietly unrepresentative.
    ds = load_dataset(spec["hf_id"], split=spec["split"]).shuffle(seed=seed)

    samples, skipped = [], 0
    for row in ds:
        if len(samples) >= n:
            break
        question, answers = _question(row), _answers(row)
        image = row.get("image")
        if not question or not answers or image is None:
            skipped += 1
            continue

        sample_id = f"{source}_{len(samples):04d}"
        image_path = images_dir / f"{sample_id}.png"
        image.convert("RGB").save(image_path)

        samples.append({
            "id": sample_id,
            # Relative when out_dir is relative (the default, resolved against the
            # repo root by load_labeled_set); absolute passes through unchanged,
            # since `root / absolute` yields the absolute path in pathlib.
            "image": str(image_path),
            "category": spec["category"],
            "prompt": question.strip() + PROMPT_SUFFIX,
            "schema": {"answer": "string"},
            "ground_truth": {"answer": {"any_of": answers}},
        })

    payload = {
        "_about": (
            f"{spec['description']}. {len(samples)} samples drawn from "
            f"{spec['hf_id']} split '{spec['split']}', shuffled with seed={seed}. "
            "Ground truth uses {'any_of': [...]} because these datasets annotate "
            "several acceptable answer spellings per question. Each sample is a "
            "single-field extraction, so this set measures extraction accuracy "
            "with statistical power but exercises multi-field schema behaviour "
            "much less than benchmarks/labeled_set.json -- run both."
        ),
        "source": spec["hf_id"],
        "split": spec["split"],
        "seed": seed,
        "n_requested": n,
        "n_built": len(samples),
        "n_skipped_incomplete": skipped,
        "samples": samples,
    }

    out_path = out_dir / "labeled_set.json"
    out_path.write_text(json.dumps(payload, indent=2))
    print(
        f"[dataset] {len(samples)} samples ({skipped} skipped for missing "
        f"question/answer/image) -> {out_path}",
        file=sys.stderr,
    )
    return out_path


def main(argv: Optional[list[str]] = None):
    import argparse

    argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(
        prog="vf dataset",
        description="Build a labeled evaluation set from a public VQA dataset",
    )
    parser.add_argument("source", choices=sorted(SOURCES),
                        help="which dataset to draw from")
    parser.add_argument("-n", type=int, default=200, help="number of samples (default 200)")
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    path = build(args.source, n=args.n, out_dir=args.out_dir, seed=args.seed)
    print(path)
    print(f"\nRun the accuracy harness against it with:\n  vf accuracy --labeled-set {path}")


if __name__ == "__main__":
    main()
