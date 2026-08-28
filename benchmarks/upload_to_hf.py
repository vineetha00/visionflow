"""Publish the evaluation dataset and benchmark results to HuggingFace.

Uploads to a *dataset* repo (`Vineetha00/visionflow`), not a model repo: VisionFlow
ships no fine-tuned weights, so there is nothing to put in a model repo. What is
worth publishing is the thing that makes the README's numbers checkable — the
synthetic images, their ground-truth labels, and the raw benchmark output.

Everything uploaded is fabricated. The sample images contain no real patient,
shipment, or company data; see examples/generate_sample_images.py for how each
value is drawn.

Requires HF_TOKEN in the environment (huggingface_hub picks it up automatically).
"""

import json
import sys
from pathlib import Path

from huggingface_hub import HfApi, create_repo

REPO_ID = "Vineetha00/visionflow"
ROOT = Path(__file__).parent.parent

DATASET_CARD = """---
license: mit
task_categories:
  - image-to-text
  - visual-question-answering
tags:
  - vlm
  - document-intelligence
  - quantization
  - benchmark
  - synthetic
---

# VisionFlow evaluation set and benchmark results

Companion data for [VisionFlow](https://github.com/vineetha00/visionflow), an
edge-first quantized VLM pipeline for local document intelligence.

## Contents

| Path | What it is |
|---|---|
| `sample_images/` | Three synthetic document images: a surgical report, a cold-chain shipping manifest, and a bar chart |
| `labeled_set.json` | Ground-truth field labels for those images |
| `results/` | Raw output from `vf bench`, `vf accuracy`, and the ONNX provider benchmark |

## Ground truth

Labels are the literal strings drawn into each image by the generator script, so
they are exact by construction rather than human-annotated. The trade-off is scope:
this is 3 images and 21 fields, sized to expose *relative* differences between
quantization levels — not to support an absolute accuracy claim.

## Synthetic data notice

Every name, MRN, date, diagnosis code, SKU, lot number, and revenue figure here is
fabricated. No real patient, shipment, or company data is present. The medical
image is a plausible-looking surgical report and is not a real medical record.

## Reproducing

```bash
pip install visionflow
vf bench
vf accuracy
```
"""


def main():
    api = HfApi()
    try:
        create_repo(REPO_ID, repo_type="dataset", exist_ok=True)
    except Exception as e:
        print(f"could not create/access {REPO_ID}: {e}", file=sys.stderr)
        return 1

    card_path = ROOT / "benchmarks" / "_README_dataset.md"
    card_path.write_text(DATASET_CARD)
    api.upload_file(
        repo_id=REPO_ID, repo_type="dataset",
        path_or_fileobj=str(card_path), path_in_repo="README.md",
    )
    card_path.unlink(missing_ok=True)

    api.upload_folder(
        repo_id=REPO_ID, repo_type="dataset",
        folder_path=str(ROOT / "examples" / "sample_images"),
        path_in_repo="sample_images",
    )

    labeled = ROOT / "benchmarks" / "labeled_set.json"
    if labeled.exists():
        api.upload_file(
            repo_id=REPO_ID, repo_type="dataset",
            path_or_fileobj=str(labeled), path_in_repo="labeled_set.json",
        )

    results_dir = ROOT / "benchmarks" / "results"
    uploaded = []
    for report in sorted(results_dir.glob("*.json")) + sorted(results_dir.glob("*.md")):
        api.upload_file(
            repo_id=REPO_ID, repo_type="dataset",
            path_or_fileobj=str(report), path_in_repo=f"results/{report.name}",
        )
        uploaded.append(report.name)

    print(f"Uploaded {len(uploaded)} result file(s): {', '.join(uploaded) or 'none'}")
    print(f"https://huggingface.co/datasets/{REPO_ID}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
