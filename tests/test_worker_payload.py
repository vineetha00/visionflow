"""The harness subprocess payload must survive sizes argv cannot carry."""

import json
from pathlib import Path

from visionflow._worker import read_payload, write_payload


def test_round_trip_through_file():
    data = {"a": 1, "samples": [{"id": i, "prompt": "x" * 50} for i in range(10)]}
    path = write_payload(data)
    assert Path(path).exists()
    assert read_payload(path) == data
    assert not Path(path).exists()          # worker cleans up after reading


def test_payload_larger_than_any_argv_limit():
    """n=500 DocVQA samples blew Linux's 128 KiB per-argument cap. Make the
    payload well past that and prove the file path is what crosses the boundary."""
    data = {"samples": [{"id": i, "prompt": "q" * 400, "schema": {"answer": "string"},
                         "ground_truth": {"answer": {"any_of": ["a" * 100]}}}
                        for i in range(1000)]}
    assert len(json.dumps(data)) > 128 * 1024
    path = write_payload(data)
    assert len(path) < 256                   # argv carries this, not the data
    assert read_payload(path)["samples"][999]["id"] == 999


def test_inline_json_still_accepted():
    assert read_payload('{"k": "v"}') == {"k": "v"}


def test_cleanup_can_be_disabled():
    path = write_payload({"x": 1})
    read_payload(path, cleanup=False)
    assert Path(path).exists()
    Path(path).unlink()
