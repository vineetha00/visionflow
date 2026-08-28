"""Constrained decoding vs. the repair loop, on the same image and schema.

Run A uses the v1 approach: generate freely, then re-prompt the model to fix the
output if it didn't parse. Run B constrains the decoder so that tokens which would
make valid JSON unreachable are masked at each step.

What to look for in the output:
  - `repaired=True` on run A means a second full forward pass was spent fixing syntax.
  - Run B should never need one, because invalid JSON is unrepresentable there.
  - The extracted *values* may still differ from ground truth in both runs.
    Constrained decoding guarantees syntax, not correctness — see README.
"""

import json
import time
from pathlib import Path

from visionflow import VisionFlow

IMAGE = Path(__file__).parent / "sample_images" / "medical_report.png"

SCHEMA = {
    "patient_name": "string",
    "mrn": "string",
    "date_of_surgery": "string (YYYY-MM-DD)",
    "procedure": "string",
    "diagnosis_code": "string",
    "estimated_blood_loss": "string",
}

PROMPT = "Extract the patient data, procedure details, and diagnosis code from this surgical report."


def run(vf, constrained: bool):
    label = "constrained decoding" if constrained else "repair loop (v1)"
    start = time.perf_counter()
    result = vf.json(IMAGE, prompt=PROMPT, schema=SCHEMA, constrained=constrained)
    elapsed = time.perf_counter() - start

    print(f"\n=== {label} ===")
    print(f"parsed ok : {result.ok}")
    print(f"repaired  : {result.repaired}")
    print(f"elapsed   : {elapsed:.1f}s")
    if result.ok:
        print(json.dumps(result.parsed, indent=2))
    else:
        print(f"error     : {result.error}")
        print(f"raw       : {result.raw_output[:400]}")
    return result


if __name__ == "__main__":
    vf = VisionFlow()
    stats = vf.load()
    print(f"Loaded {vf.engine.model_id} on {stats.hardware.value} ({stats.quantization}) "
          f"in {stats.load_seconds:.1f}s")

    run(vf, constrained=False)
    run(vf, constrained=True)
