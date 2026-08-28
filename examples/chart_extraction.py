"""Use case C: chart and dashboard understanding.

Extracts the underlying data points and a trend description from a
screenshot of a bar chart — useful for pulling numbers out of a dashboard
screenshot without re-deriving them from a data source.
"""

from pathlib import Path

from visionflow import VisionFlow

IMAGE = Path(__file__).parent / "sample_images" / "chart_dashboard.png"

SCHEMA = {
    "chart_title": "string",
    "x_axis_label": "string",
    "y_axis_label": "string",
    "data_points": [{"category": "string", "value": "number"}],
    "trend_description": "string",
}

if __name__ == "__main__":
    vf = VisionFlow()
    vf.load()

    result = vf.json(
        IMAGE,
        prompt="Extract the chart title, axis labels, every data point (category and value), and a one-sentence trend description.",
        schema=SCHEMA,
    )

    if result.ok:
        import json

        print(json.dumps(result.parsed, indent=2))
    else:
        print(f"Extraction failed: {result.error}")
