"""Generates synthetic (fully fake, non-PHI) sample images for the three demo use cases.

These are NOT real medical records or real shipping data — every name, MRN, code, and
number below is fabricated for this repo. Run once to populate examples/sample_images/.
"""

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

OUT_DIR = Path(__file__).parent / "sample_images"


def _font(size):
    for candidate in ["/System/Library/Fonts/Supplemental/Arial.ttf", "/System/Library/Fonts/Helvetica.ttc"]:
        try:
            return ImageFont.truetype(candidate, size)
        except Exception:
            continue
    return ImageFont.load_default()


def make_medical_report():
    img = Image.new("RGB", (850, 1100), "white")
    d = ImageDraw.Draw(img)
    title_font, label_font, body_font = _font(26), _font(16), _font(15)

    d.text((40, 30), "SYNTHETIC SURGICAL REPORT (fabricated sample data — not a real patient)", fill="red", font=label_font)
    d.rectangle([40, 60, 810, 110], outline="black", width=2)
    d.text((50, 70), "General Surgery Associates — Post-Operative Report", fill="black", font=title_font)

    fields = [
        ("Patient Name", "Jordan A. Rivera (SYNTHETIC)"),
        ("MRN", "SYN-00417293"),
        ("DOB", "1978-03-14"),
        ("Date of Surgery", "2026-06-02"),
        ("Surgeon", "Dr. L. Whitfield, MD"),
        ("Procedure", "Laparoscopic cholecystectomy"),
        ("Diagnosis Code (ICD-10)", "K80.20 — Cholelithiasis without obstruction"),
        ("ASA Class", "II"),
        ("Estimated Blood Loss", "25 mL"),
        ("Operative Time", "58 minutes"),
    ]
    y = 140
    for label, value in fields:
        d.text((50, y), f"{label}:", fill="black", font=label_font)
        d.text((320, y), value, fill="black", font=body_font)
        y += 34

    d.line([40, y + 10, 810, y + 10], fill="gray", width=1)
    y += 30
    d.text((50, y), "Findings:", fill="black", font=label_font)
    y += 26
    findings = [
        "Gallbladder distended with multiple gallstones, wall thickness 4mm.",
        "No evidence of common bile duct injury. Cystic duct clipped and divided.",
        "Mild adhesions to omentum, lysed without complication.",
    ]
    for line in findings:
        d.text((60, y), f"- {line}", fill="black", font=body_font)
        y += 24

    y += 20
    d.text((50, y), "Flagged Anomalies:", fill="darkred", font=label_font)
    y += 26
    d.text((60, y), "- Incidental 8mm hepatic cyst noted on segment IV; recommend follow-up ultrasound in 6 months.", fill="darkred", font=body_font)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    img.save(OUT_DIR / "medical_report.png")


def make_shipping_manifest():
    img = Image.new("RGB", (900, 700), "white")
    d = ImageDraw.Draw(img)
    title_font, label_font, body_font = _font(24), _font(15), _font(14)

    d.rectangle([30, 20, 870, 70], outline="black", width=2)
    d.text((40, 30), "COLD CHAIN SHIPPING MANIFEST (synthetic sample data)", fill="black", font=title_font)

    header = [
        ("Manifest No.", "SM-2026-0642"),
        ("Ship Date", "2026-07-10"),
        ("Origin DC", "DC-WEST-04, Fort Worth TX"),
        ("Destination DC", "DC-EAST-11, Columbus OH"),
        ("Carrier", "ColdLink Logistics"),
        ("Temp Requirement", "2-8 C (refrigerated)"),
    ]
    y = 90
    for label, value in header:
        d.text((40, y), f"{label}:", fill="black", font=label_font)
        d.text((260, y), value, fill="black", font=body_font)
        y += 28

    y += 20
    cols = ["Product Code", "Description", "Qty", "Lot", "Expiry"]
    x_positions = [40, 190, 480, 560, 700]
    for x, col in zip(x_positions, cols):
        d.text((x, y), col, fill="white", font=label_font)
    d.rectangle([30, y - 5, 870, y + 25], fill="black")
    for x, col in zip(x_positions, cols):
        d.text((x, y), col, fill="white", font=label_font)
    y += 35

    rows = [
        ("SKU-88213", "IOL implant tray, 20D", "48", "L2026-114", "2027-09-01"),
        ("SKU-88240", "Viscoelastic gel 1mL", "300", "L2026-098", "2027-05-15"),
        ("SKU-88301", "Sterile irrigation set", "120", "L2026-102", "2028-01-20"),
        ("SKU-88355", "Preloaded injector unit", "60", "L2026-119", "2027-11-30"),
    ]
    for row in rows:
        for x, val in zip(x_positions, row):
            d.text((x, y), val, fill="black", font=body_font)
        y += 30

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    img.save(OUT_DIR / "shipping_manifest.png")


def make_chart():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    quarters = ["Q1", "Q2", "Q3", "Q4"]
    revenue = [4.2, 5.1, 4.8, 6.3]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(quarters, revenue, color="#4C72B0")
    ax.set_title("Synthetic Quarterly Revenue ($M) — Sample Dashboard")
    ax.set_ylabel("Revenue ($M)")
    for i, v in enumerate(revenue):
        ax.text(i, v + 0.1, f"${v}M", ha="center")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_DIR / "chart_dashboard.png", dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    make_medical_report()
    make_shipping_manifest()
    make_chart()
    print(f"Sample images written to {OUT_DIR}")
