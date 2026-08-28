"""Use case B: supply chain document processing.

Extracts product codes, quantities, DC routing, and dates from a synthetic
cold-chain shipping manifest — the kind of document that connects directly
to real-world cold chain / DC routing workflows.
"""

from pathlib import Path

from visionflow import VisionFlow

IMAGE = Path(__file__).parent / "sample_images" / "shipping_manifest.png"

FIELDS = [
    "manifest_no",
    "ship_date",
    "origin_dc",
    "destination_dc",
    "carrier",
    "temp_requirement",
]

if __name__ == "__main__":
    vf = VisionFlow()
    vf.load()

    header = vf.key_value(
        IMAGE,
        prompt="Extract the manifest header fields from this shipping manifest.",
        fields=FIELDS,
    )
    print("Header:", header)

    line_items = vf.json(
        IMAGE,
        prompt="Extract every line item row (product code, description, quantity, lot, expiry) as a JSON list under a 'line_items' key.",
        schema={"line_items": [{"product_code": "string", "description": "string", "qty": "number", "lot": "string", "expiry": "string"}]},
    )
    if line_items.ok:
        import json

        print("Line items:", json.dumps(line_items.parsed, indent=2))
    else:
        print(f"Line item extraction failed: {line_items.error}")
