"""Payload hand-off between a harness and the subprocess that measures one config.

Both `bench.py` and `accuracy.py` run each configuration in a fresh subprocess
(for memory isolation; see their docstrings). The subprocess needs its inputs,
and the first version passed them as one JSON string in argv.

That has a ceiling. Linux caps a single argv entry at MAX_ARG_STRLEN (128 KiB),
and the accuracy payload carries every sample's prompt, schema and ground truth.
At n=150 it fit; at n=500 it raised `OSError: [Errno 7] Argument list too long`
and the sweep died before measuring anything. The payload now goes through a
temp file and argv carries only the path. `read_payload` still accepts inline
JSON, so an old-style invocation keeps working.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path


def write_payload(data: dict) -> str:
    """Serialize `data` to a temp file; returns the path to hand to the worker."""
    fd, path = tempfile.mkstemp(prefix="visionflow-worker-", suffix=".json")
    with os.fdopen(fd, "w") as f:
        json.dump(data, f)
    return path


def read_payload(arg: str, cleanup: bool = True) -> dict:
    """Load the payload a worker was given: a file path, or inline JSON.

    The worker owns the temp file once it has read it, so by default it is
    removed here rather than left for the parent to remember.
    """
    stripped = arg.lstrip()
    if stripped.startswith("{"):
        return json.loads(arg)
    path = Path(arg)
    data = json.loads(path.read_text())
    if cleanup:
        try:
            path.unlink()
        except OSError:
            pass
    return data
