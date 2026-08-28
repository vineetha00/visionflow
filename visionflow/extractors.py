"""Prompt templates and structured-output extraction with a JSON repair step.

This is the layer that makes raw VLM text output usable in downstream pipelines:
a generation function is wrapped with schema instructions, and malformed JSON is
sent back through the model once with an explicit repair prompt before giving up.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Optional


GenerateFn = Callable[[str], str]


JSON_MODE_INSTRUCTIONS = """Respond with a single JSON object only — no prose, no markdown
code fences, no explanation before or after. If a field is not present in the image,
use null rather than omitting it or guessing.

{schema_block}

User request: {prompt}"""

KEY_VALUE_INSTRUCTIONS = """Extract the requested information as plain "Key: Value" lines,
one per line, no other text. If a field is not present in the image, write "Key: null".

Fields to extract: {fields}

User request: {prompt}"""

REPAIR_INSTRUCTIONS = """The following text was supposed to be a single valid JSON object but
failed to parse with error: {error}

Text:
{broken}

Return ONLY the corrected, valid JSON object. No prose, no markdown fences."""


@dataclass
class ExtractionResult:
    mode: str
    raw_output: str
    parsed: Any = None
    repaired: bool = False
    ok: bool = False
    error: Optional[str] = None
    constrained: bool = False


def _strip_code_fences(text: str) -> str:
    text = text.strip()
    fenced = re.match(r"^```(?:json)?\s*(.*?)\s*```$", text, re.DOTALL)
    if fenced:
        return fenced.group(1).strip()
    return text


def _extract_first_json_object(text: str) -> Optional[str]:
    """Best-effort scan for the first balanced {...} block in free-form text."""
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


def build_json_prompt(prompt: str, schema: Optional[dict] = None) -> str:
    schema_block = (
        f"Required JSON schema (fields and types):\n{json.dumps(schema, indent=2)}"
        if schema
        else "Return a JSON object with whatever fields are relevant to the request below."
    )
    return JSON_MODE_INSTRUCTIONS.format(schema_block=schema_block, prompt=prompt)


def build_key_value_prompt(prompt: str, fields: list[str]) -> str:
    return KEY_VALUE_INSTRUCTIONS.format(fields=", ".join(fields), prompt=prompt)


def parse_key_value(text: str) -> dict:
    result = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or ":" not in line:
            continue
        key, _, value = line.partition(":")
        key, value = key.strip(), value.strip()
        if value.lower() == "null":
            value = None
        result[key] = value
    return result


def extract_json(
    raw_output: str,
    generate_fn: Optional[GenerateFn] = None,
    max_repair_attempts: int = 1,
) -> ExtractionResult:
    """Parse `raw_output` as JSON, attempting cleanup and (optionally) a model-assisted
    repair pass if parsing fails.

    `generate_fn` is a callable that takes a repair prompt string and returns the
    model's text response — pass `None` to disable the repair pass (parse-only mode,
    used in tests).
    """
    candidate = _strip_code_fences(raw_output)
    try:
        parsed = json.loads(candidate)
        return ExtractionResult(mode="json", raw_output=raw_output, parsed=parsed, ok=True)
    except json.JSONDecodeError:
        pass

    extracted = _extract_first_json_object(candidate)
    if extracted:
        try:
            parsed = json.loads(extracted)
            return ExtractionResult(mode="json", raw_output=raw_output, parsed=parsed, ok=True)
        except json.JSONDecodeError as e:
            last_error = str(e)
    else:
        last_error = "no JSON object found in output"

    if generate_fn is None or max_repair_attempts < 1:
        return ExtractionResult(
            mode="json", raw_output=raw_output, ok=False, error=last_error
        )

    broken = extracted or candidate
    for _ in range(max_repair_attempts):
        repair_prompt = REPAIR_INSTRUCTIONS.format(error=last_error, broken=broken)
        repaired_text = generate_fn(repair_prompt)
        repaired_clean = _strip_code_fences(repaired_text)
        try:
            parsed = json.loads(repaired_clean)
            return ExtractionResult(
                mode="json", raw_output=raw_output, parsed=parsed, ok=True, repaired=True
            )
        except json.JSONDecodeError as e:
            last_error = str(e)
            broken = repaired_clean

    return ExtractionResult(
        mode="json", raw_output=raw_output, ok=False, error=last_error, repaired=True
    )
