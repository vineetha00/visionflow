"""Tests for the ONNX report layer.

The risk this file guards against is specific: ONNX Runtime falls back silently.
Requesting `TensorrtExecutionProvider` on a machine without it returns a working
CPU session and no error, so a naive harness would publish CPU latency under a
TensorRT heading. These tests pin the behaviour that prevents that.

No model is exported here — export is verified separately by
`python -m visionflow.onnx_export --verify`, which needs to download weights.
"""

from visionflow.onnx_export import PROVIDER_PREFERENCE, ExportResult, OnnxBenchResult, to_markdown


def test_provider_preference_puts_tensorrt_first_and_cpu_last():
    assert PROVIDER_PREFERENCE[0] == "TensorrtExecutionProvider"
    assert PROVIDER_PREFERENCE[-1] == "CPUExecutionProvider"


def test_finalize_computes_percentiles():
    r = OnnxBenchResult(provider_requested="CPUExecutionProvider")
    r.latencies_ms = [10.0, 20.0, 30.0, 40.0]
    r.finalize()
    assert r.n_runs == 4
    assert r.p50_ms == 25.0
    assert r.mean_ms == 25.0


def test_silent_fallback_is_flagged_in_the_table():
    """A TensorRT row that actually ran on CPU must say so, loudly."""
    bench = OnnxBenchResult(
        provider_requested="TensorrtExecutionProvider",
        provider_used="CPUExecutionProvider",
        ok=True,
        reason="ORT fell back to CPUExecutionProvider; this is NOT a TensorrtExecutionProvider measurement",
    )
    bench.latencies_ms = [5.0]
    bench.finalize()
    table = to_markdown(None, [bench])
    assert "NOT a TensorrtExecutionProvider measurement" in table
    assert "⚠️" in table


def test_unavailable_provider_renders_as_not_measured():
    bench = OnnxBenchResult(
        provider_requested="CUDAExecutionProvider",
        ok=False,
        reason="CUDAExecutionProvider not available in this onnxruntime build",
    )
    table = to_markdown(None, [bench])
    assert "not measured" in table
    assert "0.00" not in table


def test_export_success_reports_numeric_agreement_with_torch():
    export = ExportResult(
        ok=True, path="x.onnx", opset=17, seconds=3.2,
        input_shape=[1, 3, 384, 384], output_shape=[1, 729, 1152],
        max_abs_diff_vs_torch=1.2e-5, exporter="torchscript",
    )
    table = to_markdown(export, [])
    assert "ok" in table and "1.20e-05" in table
    # Which exporter produced the graph matters: the two disagree on this model,
    # so a reader has to be able to tell which one the numbers came from.
    assert "torchscript" in table


def test_graph_that_exports_but_cannot_run_is_not_reported_as_success():
    """The dynamo exporter emits a loadable graph for SmolVLM's vision tower that
    throws at runtime. Reporting that as a successful export would be a lie."""
    export = ExportResult(
        ok=False, opset=17,
        error="dynamo: exported but failed verification: InvalidArgument: GatherND invalid index",
    )
    table = to_markdown(export, [])
    assert "failed" in table
    assert "GatherND" in table


def test_export_failure_is_reported_not_swallowed():
    table = to_markdown(ExportResult(ok=False, error="unsupported op"), [])
    assert "failed" in table and "unsupported op" in table
