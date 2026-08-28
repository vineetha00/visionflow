"""Use case A: medical document intelligence.

Extracts structured fields from a synthetic surgical report — patient data,
diagnosis codes, measurements, and flagged anomalies — entirely on-device.
No image or extracted data ever leaves the machine.
"""

from pathlib import Path

from visionflow import VisionFlow

IMAGE = Path(__file__).parent / "sample_images" / "medical_report.png"

SCHEMA = {
    "patient_name": "string",
    "mrn": "string",
    "date_of_surgery": "string (YYYY-MM-DD)",
    "surgeon": "string",
    "procedure": "string",
    "diagnosis_code": "string (ICD-10 code and description)",
    "estimated_blood_loss": "string",
    "operative_time": "string",
    "flagged_anomalies": "list of strings",
}

if __name__ == "__main__":
    vf = VisionFlow()
    vf.load()

    result = vf.json(
        IMAGE,
        prompt="Extract the patient data, procedure details, diagnosis code, and any flagged anomalies from this surgical report.",
        schema=SCHEMA,
    )

    if result.ok:
        import json

        print(json.dumps(result.parsed, indent=2))
    else:
        print(f"Extraction failed: {result.error}")
