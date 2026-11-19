"""Tests for pydantic schema validation and the schema-aware repair path.

Constrained decoding guarantees the output parses. These cover the layer that
decides whether what parsed is the object that was actually requested — and the
repair prompt that failure produces, which is the practical payoff: naming the
bad field beats quoting a JSON parser's column number.
"""

import pytest

from visionflow.extractors import (
    REPAIR_INSTRUCTIONS,
    SCHEMA_REPAIR_INSTRUCTIONS,
    extract_json,
)
from visionflow.validation import (
    model_from_schema,
    python_type_for_hint,
    resolve_model,
    schema_dict_for_prompt,
    validate,
)

SCHEMA = {"mrn": "string", "qty": "integer", "flags": "list of strings", "urgent": "boolean"}


@pytest.fixture
def validator():
    model = model_from_schema(SCHEMA)
    return lambda obj: validate(obj, model)


@pytest.mark.parametrize(
    "hint,expected",
    [
        ("string", str),
        ("string (YYYY-MM-DD)", str),
        ("integer", int),
        ("number", float),
        ("boolean", bool),
        ("list of strings", list),
        ("array", list),
        ("ICD-10 code and description", str),  # unrecognised prose -> str
    ],
)
def test_hint_type_inference(hint, expected):
    assert python_type_for_hint(hint) is expected


def test_valid_object_passes(validator):
    outcome = validator({"mrn": "SYN-1", "qty": 48, "flags": ["x"], "urgent": True})
    assert outcome.ok and outcome.summary == ""


def test_types_are_coerced_by_default(validator):
    """A VLM reading a document returns "48" far more often than 48. Rejecting
    that would make validation an obstacle rather than a safety net."""
    outcome = validator({"mrn": "SYN-1", "qty": "48", "flags": [], "urgent": "true"})
    assert outcome.ok
    assert outcome.validated["qty"] == 48 and outcome.validated["urgent"] is True


def test_strict_mode_refuses_coercion():
    model = model_from_schema(SCHEMA)
    outcome = validate({"mrn": "SYN-1", "qty": "48", "flags": [], "urgent": True},
                       model, strict=True)
    assert not outcome.ok and "qty" in outcome.summary


def test_explicit_null_is_valid_but_a_missing_key_is_not(validator):
    """The prompt tells the model to emit null for absent fields, and the accuracy
    harness counts an object as schema-valid only when every key is present. So
    these two cases must not be conflated."""
    assert validator({"mrn": None, "qty": None, "flags": None, "urgent": None}).ok

    outcome = validator({"mrn": "SYN-1"})
    assert not outcome.ok
    assert set(outcome.missing_fields) == {"qty", "flags", "urgent"}


def test_non_object_is_rejected_with_a_readable_message(validator):
    outcome = validator([1, 2, 3])
    assert not outcome.ok and "expected a JSON object" in outcome.summary


def test_error_summary_names_fields_not_parser_columns(validator):
    outcome = validator({"mrn": "SYN-1", "qty": "forty-eight", "flags": [], "urgent": True})
    assert outcome.summary.startswith("qty:")
    assert "line" not in outcome.summary and "column" not in outcome.summary


def test_pydantic_model_accepted_as_schema():
    from pydantic import BaseModel

    class Report(BaseModel):
        patient_name: str
        qty: int

    assert resolve_model(Report) is Report
    # And it flattens back to hints so prompt, grammar and validator agree on order.
    assert schema_dict_for_prompt(Report) == {"patient_name": "string", "qty": "integer"}


def test_schema_dict_round_trips_field_order():
    assert list(schema_dict_for_prompt(SCHEMA)) == list(SCHEMA)


# --- repair-path selection -------------------------------------------------


def test_schema_failure_uses_the_schema_repair_template(validator):
    prompts = []

    def fake_model(prompt):
        prompts.append(prompt)
        return '{"mrn": "SYN-1", "qty": 48, "flags": [], "urgent": true}'

    result = extract_json('{"mrn": "SYN-1", "qty": "forty-eight", "flags": [], "urgent": true}',
                          generate_fn=fake_model, validator=validator)
    assert result.ok and result.repaired and result.schema_valid
    assert result.validated["qty"] == 48
    # It parsed fine, so telling the model it is malformed would send it hunting
    # for a syntax error that does not exist.
    assert SCHEMA_REPAIR_INSTRUCTIONS.splitlines()[0] in prompts[0]
    assert "failed to parse" not in prompts[0]
    assert "qty" in prompts[0]


def test_syntax_failure_still_uses_the_syntax_template(validator):
    prompts = []

    def fake_model(prompt):
        prompts.append(prompt)
        return '{"mrn": "SYN-1", "qty": 48, "flags": [], "urgent": true}'

    result = extract_json('{"mrn": "SYN-1", ', generate_fn=fake_model, validator=validator)
    assert result.ok
    assert REPAIR_INSTRUCTIONS.splitlines()[0] in prompts[0]


def test_validation_failure_without_repair_reports_errors(validator):
    result = extract_json('{"mrn": "SYN-1"}', validator=validator)
    assert not result.ok and result.schema_valid is False
    assert {e["loc"][0] for e in result.validation_errors} == {"qty", "flags", "urgent"}
    # The parsed object is still returned, so a caller can salvage partial fields.
    assert result.parsed == {"mrn": "SYN-1"}


def test_no_validator_leaves_schema_valid_unset():
    """`None` must stay distinguishable from False: "not checked" is not "failed"."""
    result = extract_json('{"anything": 1}')
    assert result.ok and result.schema_valid is None


def test_repair_exhaustion_returns_last_errors(validator):
    result = extract_json('{"mrn": "SYN-1"}',
                          generate_fn=lambda _p: '{"mrn": "SYN-1"}',
                          max_repair_attempts=2, validator=validator)
    assert not result.ok and result.repaired and result.schema_valid is False
    assert result.validation_errors
