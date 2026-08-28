"""Reproduce every number in the README, in one command.

    python benchmarks/run_benchmarks.py

This is a thin orchestrator over three harnesses, each of which is also usable
on its own:

  vf bench      latency / throughput / memory per backend  (visionflow/bench.py)
  vf accuracy   schema validity + field accuracy per quant (visionflow/accuracy.py)
  eval.py       GPT-4o Vision field-match baseline, if OPENAI_API_KEY is set

Latency and memory measurement used to live in this file. It was moved into
`visionflow/bench.py` because measuring several backends inside one process makes
the memory numbers meaningless — the second backend measured inherits whatever the
first one allocated. `bench.py` runs each configuration in its own subprocess.
The old in-process path is gone rather than deprecated, so nobody can accidentally
publish a number it produced.

The GPT-4o comparison is opt-in and network-dependent by nature: it uploads the
sample images to OpenAI. Those images are synthetic and contain no real patient or
shipment data, but note that this is the one part of this repo that leaves the
machine, and it stays off unless you set the key.
"""

import json
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from visionflow.eval import compare_to_gpt4o, save_report  # noqa: E402
from visionflow.pipeline import VisionFlow  # noqa: E402

SAMPLE_DIR = ROOT / "examples" / "sample_images"
RESULTS = Path(__file__).parent / "results"


def _run(module_args: list[str], label: str) -> int:
    print(f"\n{'=' * 70}\n=== {label}\n{'=' * 70}", flush=True)
    proc = subprocess.run([sys.executable, "-m", *module_args], cwd=str(ROOT))
    if proc.returncode != 0:
        print(f"!! {label} exited {proc.returncode}", file=sys.stderr)
    return proc.returncode


def run_gpt4o_comparison():
    """Field-match rate of local extraction vs GPT-4o Vision on the same images."""
    print(f"\n{'=' * 70}\n=== GPT-4o Vision comparison\n{'=' * 70}", flush=True)

    labeled = json.loads((Path(__file__).parent / "labeled_set.json").read_text())
    samples = [
        {
            "image_path": str(ROOT / s["image"]),
            "prompt": s["prompt"],
            "schema": s["schema"],
        }
        for s in labeled["samples"]
    ]

    import os

    if not os.environ.get("OPENAI_API_KEY"):
        # Don't pay the cost of loading a multi-GB VLM just to discover the
        # baseline can't run.
        result = compare_to_gpt4o(samples, vf_generate=None)
        print(f"  not run: {result.reason}")
        save_report(asdict(result), str(RESULTS / "gpt4o_comparison.json"))
        return

    vf = VisionFlow()
    vf.load()
    result = compare_to_gpt4o(samples, lambda p, prompt, schema: vf.json(p, prompt, schema=schema))
    if result.field_match_rate is not None:
        print(f"  field match rate vs GPT-4o: {result.field_match_rate:.1%}")
    else:
        print(f"  no comparable fields ({result.reason})")
    save_report(asdict(result), str(RESULTS / "gpt4o_comparison.json"))


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)

    _run(["visionflow.bench", "--repeats", "3", "--max-new-tokens", "256"],
         "Latency / throughput / memory  (vf bench)")
    _run(["visionflow.accuracy"],
         "Accuracy per quantization level  (vf accuracy)")
    _run(["visionflow.onnx_export", "--verify", "--benchmark"],
         "ONNX export + execution providers")
    run_gpt4o_comparison()

    print(f"\nAll reports written to {RESULTS}")


if __name__ == "__main__":
    main()
