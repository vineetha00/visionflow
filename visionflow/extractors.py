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

# A separate template for schema failures. The JSON parsed fine here, so telling
# the model it is malformed would be wrong and would send it looking for a syntax
# error that does not exist. What it needs is the list of offending fields.
SCHEMA_REPAIR_INSTRUCTIONS = """The following JSON parsed correctly but does not match the
requested schema: {error}

JSON:
{broken}

Fix only the fields named above. Keep every other field exactly as it is. Use null for a
field whose value is genuinely not present in the image — do not invent one. Return ONLY
the corrected JSON object, with no prose and no markdown fences."""


@dataclass
class ExtractionResult:
    mode: str
    raw_output: str
    parsed: Any = None
    repaired: bool = False
    ok: bool = False
    error: Optional[str] = None
    constrained: bool = False
    # Token accounting, so the constrained-vs-repair speedup can be decomposed.
    # "Constrained decoding is faster" has two possible causes -- it skips the
    # repair pass, and the grammar ends generation when the object closes rather
    # than running toward the token limit. Latency alone cannot separate them.
    new_tokens: int = 0        # tokens from the first generation
    repair_tokens: int = 0     # tokens spent on repair passes
    forward_passes: int = 1    # 1, or 1 + number of repair attempts made
    # Schema validation (pydantic). `schema_valid` is None when no schema was
    # given, so "not checked" stays distinguishable from "checked and failed".
    validated: Any = None
    schema_valid: Optional[bool] = None
    validation_errors: list = field(default_factory=list)
    # Schema-structural speculation, when it ran. See speculative.py.
    speculation: Optional[dict] = None

    @property
    def total_tokens(self) -> int:
        return self.new_tokens + self.repair_tokens


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
    validator: Optional[Callable[[Any], Any]] = None,
) -> ExtractionResult:
    """Parse `raw_output` as JSON, with optional schema validation and a
    model-assisted repair pass.

    `generate_fn` takes a repair prompt and returns the model's response — pass
    `None` to disable repair (parse-only, used in tests).

    `validator` takes the parsed object and returns a `ValidationOutcome`. It
    turns the repair loop from a syntax fixer into a schema fixer: without it,
    the best a repair prompt can say is what the JSON parser said ("Expecting
    ',' delimiter: line 1 column 92"), which tells the model nothing about the
    extraction task. With it, the prompt names the offending fields.
    """
    candidate = _strip_code_fences(raw_output)
    parsed, last_error = None, None

    for text in (candidate, _extract_first_json_object(candidate)):
        if not text:
            continue
        try:
            parsed = json.loads(text)
            break
        except json.JSONDecodeError as e:
            last_error = str(e)
    if parsed is None and last_error is None:
        last_error = "no JSON object found in output"

    def _finish(obj, repaired: bool) -> Optional[ExtractionResult]:
        """Validate `obj`; return a result if it is acceptable, else None."""
        if validator is None:
            return ExtractionResult(mode="json", raw_output=raw_output, parsed=obj,
                                    ok=True, repaired=repaired)
        outcome = validator(obj)
        if outcome.ok:
            return ExtractionResult(
                mode="json", raw_output=raw_output, parsed=obj, ok=True, repaired=repaired,
                validated=outcome.validated, schema_valid=True,
            )
        return None

    if parsed is not None:
        done = _finish(parsed, repaired=False)
        if done is not None:
            return done
        outcome = validator(parsed)
        last_error = outcome.summary
        schema_failure = True
    else:
        schema_failure = False

    if generate_fn is None or max_repair_attempts < 1:
        return ExtractionResult(
            mode="json", raw_output=raw_output, parsed=parsed,
            ok=parsed is not None and validator is None,
            error=last_error,
            validation_errors=(validator(parsed).errors if validator and parsed is not None else []),
            schema_valid=False if validator is not None else None,
        )

    broken = json.dumps(parsed) if parsed is not None else (
        _extract_first_json_object(candidate) or candidate
    )
    for _ in range(max_repair_attempts):
        template = SCHEMA_REPAIR_INSTRUCTIONS if schema_failure else REPAIR_INSTRUCTIONS
        repaired_text = generate_fn(template.format(error=last_error, broken=broken))
        repaired_clean = _strip_code_fences(repaired_text)
        try:
            candidate_obj = json.loads(repaired_clean)
        except json.JSONDecodeError as e:
            last_error, broken, schema_failure = str(e), repaired_clean, False
            continue
        done = _finish(candidate_obj, repaired=True)
        if done is not None:
            return done
        outcome = validator(candidate_obj)
        last_error, broken, schema_failure = outcome.summary, repaired_clean, True
        parsed = candidate_obj

    return ExtractionResult(
        mode="json", raw_output=raw_output, parsed=parsed, ok=False, error=last_error,
        repaired=True,
        validation_errors=(validator(parsed).errors if validator and parsed is not None else []),
        schema_valid=False if validator is not None else None,
    )
