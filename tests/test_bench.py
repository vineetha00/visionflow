"""Tests for the benchmark and accuracy harnesses' reporting logic.

These cover the parts that decide whether a published number is honest: percentile
maths, the skip path, and the markdown renderer's handling of missing measurements.
No model is loaded.
"""

import json

import pytest

from visionflow.accuracy import AccuracyResult, normalize, score_field
from visionflow.accuracy import to_markdown as accuracy_markdown
from visionflow.bench import BenchResult, _parse_worker_output, _percentile, to_markdown


def test_percentile_endpoints_and_interpolation():
    data = [1.0, 2.0, 3.0, 4.0]
    assert _percentile(data, 0) == 1.0
    assert _percentile(data, 100) == 4.0
    assert _percentile(data, 50) == pytest.approx(2.5)
    assert _percentile([5.0], 95) == 5.0


def test_finalize_computes_tokens_per_second_over_totals():
    r = BenchResult(label="x", backend="cpu", model_id="m")
    r.latencies = [1.0, 3.0]
    r.tokens = [10, 30]
    r.finalize()
    # Aggregate throughput (40 tokens / 4s), not the mean of per-run rates —
    # averaging rates would over-weight short runs.
    assert r.tokens_per_second == pytest.approx(10.0)
    assert r.n_measured == 2
    assert r.p50_s == pytest.approx(2.0)


def test_finalize_is_safe_with_no_measurements():
    r = BenchResult(label="x", backend="cuda", model_id="m", skipped=True, reason="no CUDA")
    r.finalize()
    assert r.mean_s is None and r.tokens_per_second is None


def test_markdown_renders_skips_as_not_measured():
    report = {
        "methodology": {"n_images": 3, "repeats_per_image": 3, "process_isolation": True,
                        "warmup_runs_excluded": 1, "max_new_tokens": 256},
        "host": {"platform": "test", "processor": "test", "total_ram_gb": 16, "torch": "2.13"},
        "results": [
            {"label": "CUDA / INT4", "backend": "cuda", "model_id": "org/M",
             "skipped": True, "reason": "no CUDA device on this machine"},
        ],
    }
    table = to_markdown(report)
    assert "not measured" in table
    assert "no CUDA device" in table
    # A skipped row must never render a numeric-looking cell.
    assert "0.00" not in table


def test_markdown_includes_isolation_caveat_when_disabled():
    report = {"methodology": {"process_isolation": False}, "host": {}, "results": []}
    assert "memory figures not independent" in to_markdown(report)


def test_worker_output_parsing_ignores_progress_noise():
    payload = BenchResult(label="mps", backend="mps", model_id="org/M")
    from dataclasses import asdict

    stdout = (
        "Loading weights:  50%|#####     | 1/2\n"
        "some warning from transformers\n"
        + "__VF_BENCH_JSON__" + json.dumps(asdict(payload)) + "\n"
    )
    parsed = _parse_worker_output(stdout)
    assert parsed is not None and parsed.backend == "mps"


def test_worker_output_parsing_returns_none_without_marker():
    assert _parse_worker_output("crashed\ntraceback\n") is None


# --- accuracy scoring -------------------------------------------------------


def test_normalize_folds_case_whitespace_and_unicode_dashes():
    assert normalize("  K80.20 — Cholelithiasis  ") == normalize("k80.20 - cholelithiasis")


def test_normalize_preserves_digits_and_punctuation_that_carry_meaning():
    # An MRN differing by one digit must not normalize to equality.
    assert normalize("SYN-00417293") != normalize("SYN-00417294")


def test_normalize_joins_list_values():
    assert normalize(["a", "b"]) == "a b"


def test_score_field_exact_fuzzy_and_miss():
    exact, fuzzy, sim = score_field("25 mL", "25 ml")
    assert exact and fuzzy and sim == 1.0

    exact, fuzzy, sim = score_field("SYN-00417293", "SYN-0041729")
    assert not exact and fuzzy and sim > 0.85

    exact, fuzzy, sim = score_field("Laparoscopic cholecystectomy", "chest x-ray")
    assert not exact and not fuzzy and sim < 0.85


def test_score_field_treats_missing_value_as_failure():
    assert score_field("something", None) == (False, False, pytest.approx(0.0, abs=1e-9))


def test_accuracy_rates_are_none_before_any_samples():
    r = AccuracyResult(label="x", backend="cpu", model_id="m", constrained=False)
    assert r.schema_validity_rate is None and r.exact_accuracy is None


def test_is_built_requires_images_not_just_the_manifest(tmp_path):
    """The manifest is committed and the images are gitignored, so a fresh clone
    has the JSON and none of the PNGs. A check that trusted the manifest alone
    let a cluster job skip the build and fail on missing files hours later."""
    import json as _json

    from visionflow.datasets import is_built

    manifest = tmp_path / "labeled_set.json"
    assert not is_built(manifest)                       # absent entirely

    manifest.write_text(_json.dumps(
        {"samples": [{"image": str(tmp_path / "images" / "a.png")}]}))
    assert not is_built(manifest)                       # manifest, no images

    (tmp_path / "images").mkdir()
    (tmp_path / "images" / "a.png").write_bytes(b"x")
    assert is_built(manifest)                           # both present

    manifest.write_text("not json")
    assert not is_built(manifest)                       # corrupt

    manifest.write_text(_json.dumps({"samples": []}))
    assert not is_built(manifest)                       # empty


def test_wilson_ci_bounds_are_sane():
    from visionflow.accuracy import wilson_ci

    assert wilson_ci(0, 0) == (0.0, 0.0)
    lo, hi = wilson_ci(21, 21)
    assert hi > 0.999 and 0.8 < lo < 1.0        # never exceeds 1 even at p=1
    lo, hi = wilson_ci(0, 21)
    assert lo == 0.0 and hi < 0.2               # never below 0 even at p=0
    lo, hi = wilson_ci(8, 21)
    assert lo < 8 / 21 < hi


def test_overlap_detects_indistinguishable_configurations():
    """The differences this project reports are small and the sample is 21 fields.
    These are the actual measured counts: INT4 (8/21) vs INT8 (6/21) must come out
    indistinguishable, while fp32 (16/21) vs fp16-on-MPS (1/21) must not."""
    from visionflow.accuracy import overlaps, wilson_ci

    assert overlaps(wilson_ci(8, 21), wilson_ci(6, 21))
    assert overlaps(wilson_ci(16, 21), wilson_ci(8, 21))     # fp32 vs INT4: marginal
    assert not overlaps(wilson_ci(16, 21), wilson_ci(1, 21))  # fp32 vs MPS fp16: real


def test_accuracy_markdown_flags_overlapping_configurations():
    report = {
        "methodology": {"n_samples": 3, "n_fields": 21, "fuzzy_threshold": 0.85},
        "results": [
            {"label": "INT4", "quantization": "int4", "n_samples": 3, "n_fields": 21,
             "n_exact": 8, "n_fuzzy": 12, "n_schema_valid": 2, "n_repaired": 2,
             "exact_accuracy": 8 / 21, "exact_accuracy_ci": [0.21, 0.59],
             "fuzzy_accuracy": 12 / 21, "fuzzy_accuracy_ci": [0.36, 0.75],
             "schema_validity_rate": 2 / 3, "schema_validity_ci": [0.21, 0.94],
             "mean_seconds": 4.3},
            {"label": "INT8", "quantization": "int8", "n_samples": 3, "n_fields": 21,
             "n_exact": 6, "n_fuzzy": 6, "n_schema_valid": 1, "n_repaired": 2,
             "exact_accuracy": 6 / 21, "exact_accuracy_ci": [0.14, 0.50],
             "fuzzy_accuracy": 6 / 21, "fuzzy_accuracy_ci": [0.14, 0.50],
             "schema_validity_rate": 1 / 3, "schema_validity_ci": [0.06, 0.79],
             "mean_seconds": 11.5},
        ],
    }
    table = accuracy_markdown(report)
    assert "95% Wilson" in table
    assert "[21–59]" in table
    # The table must say out loud that the leader is not separable from INT8,
    # rather than leaving a reader to rank on point estimates alone.
    assert "Not statistically distinguishable" in table
    assert "INT8" in table.split("Not statistically distinguishable")[1]


def test_accuracy_markdown_reports_skip_reason():
    report = {
        "methodology": {"n_samples": 3, "n_fields": 21, "fuzzy_threshold": 0.85},
        "results": [{"label": "CUDA / INT4", "skipped": True, "reason": "no CUDA device"}],
    }
    table = accuracy_markdown(report)
    assert "not measured" in table and "no CUDA device" in table
