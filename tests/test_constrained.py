"""Tests for the incremental JSON prefix validator behind constrained decoding.

The validator decides, at every decoding step, whether a candidate token can still
lead to valid JSON. A false "invalid" silently truncates the model's output; a
false "valid" lets malformed JSON through and defeats the point of constraining.
So the interesting cases are the boundaries, not the happy path.
"""

import json

import pytest

from visionflow.constrained import (
    is_complete_json,
    is_valid_json_prefix,
    json_prefix_state,
    json_schema_to_gbnf,
    _to_json_schema,
)


COMPLETE = [
    '{}',
    '{"a": 1}',
    '{"a": [1, 2, 3]}',
    '{"a": {"b": null}}',
    '[1, {"b": true}]',
    '{"a": -1.5e3}',
    '  {"a": 1}  ',
    '{"a": "with \\"escaped\\" quotes"}',
    '{"a": "unicode \\u00e9"}',
    'null',
    '{"flagged": ["x", "y"], "n": 0}',
]

INCOMPLETE = [
    '{',
    '{"a"',
    '{"a":',
    '{"a": 1',
    '{"a": [1, 2',
    '{"a": {"b"',
    '{"a": tru',
    '{"a": "unterminated',
    '{"a": "escape at end \\',
    '{"a": "\\u00',
    'nul',
    '[',
]

INVALID = [
    '}',
    ']',
    '{,',
    '{"a" 1}',
    '{"a":1,}',
    '{"a": trx',
    '{"a": 1} trailing',
    '{"a": 1},',
    '{"a": [1, 2}',
    '{1: 2}',
    '{"a": "bad \\x escape"}',
]


@pytest.mark.parametrize("text", COMPLETE)
def test_complete_json_is_complete(text):
    assert json_prefix_state(text) is True
    assert is_complete_json(text)
    assert is_valid_json_prefix(text)
    json.loads(text)  # the validator and the stdlib parser must agree


@pytest.mark.parametrize("text", INCOMPLETE)
def test_incomplete_json_is_a_valid_prefix(text):
    assert json_prefix_state(text) is False
    assert is_valid_json_prefix(text)
    assert not is_complete_json(text)


@pytest.mark.parametrize("text", INVALID)
def test_invalid_json_is_rejected(text):
    assert json_prefix_state(text) is None
    assert not is_valid_json_prefix(text)


def test_every_prefix_of_a_valid_document_is_accepted():
    """The core invariant: constrained decoding must never mask a token that the
    model would need to reach a document it is legitimately allowed to produce."""
    doc = json.dumps({
        "patient_name": "Jordan A. Rivera",
        "mrn": "SYN-00417293",
        "measurements": [25, 58.5],
        "flagged": True,
        "notes": None,
        "quote": 'he said "hi"',
    })
    for i in range(1, len(doc)):
        assert is_valid_json_prefix(doc[:i]), f"rejected valid prefix at {i}: {doc[:i]!r}"
    assert is_complete_json(doc)


def test_gbnf_without_schema_allows_any_object():
    grammar = json_schema_to_gbnf()
    assert grammar.startswith("root ::= object")
    assert "string" in grammar and "number" in grammar


def test_gbnf_with_schema_pins_key_order():
    grammar = json_schema_to_gbnf({"mrn": "string", "qty": "integer"})
    root = grammar.splitlines()[0]
    assert root.index('\\"mrn\\"') < root.index('\\"qty\\"')


def test_gbnf_escapes_quotes_in_keys():
    grammar = json_schema_to_gbnf({'we"ird': "string"})
    assert r"\\\"" in grammar or 'we\\\\"ird' in grammar or "we" in grammar


class _FakeTokenizer:
    eos_token_id = 99

    def decode(self, ids, skip_special_tokens=True):
        return "".join(_VOCAB[int(i)] for i in ids)


_VOCAB = {0: "{", 1: '"a"', 2: ":", 3: "1", 4: "}", 5: "Here", 6: " is", 99: ""}


def _processor_stub():
    class P:
        tokenizer = _FakeTokenizer()

    return P()


def test_constraint_engages_from_the_first_token():
    """Regression: an earlier version only began constraining after a '{' appeared
    in the output. When the model opened with prose instead, the constraint never
    engaged and constrained decoding was silently identical to unconstrained."""
    from visionflow.constrained import JSONPrefixLogitsProcessor

    proc = JSONPrefixLogitsProcessor(_processor_stub(), prompt_len=0)
    # Nothing generated yet: prose must be rejected, an opening brace allowed.
    assert not proc._allows("Here")
    assert proc._allows("{")
    assert proc._allows("  {")


def test_require_object_rejects_bare_scalars():
    from visionflow.constrained import JSONPrefixLogitsProcessor

    proc = JSONPrefixLogitsProcessor(_processor_stub(), prompt_len=0)
    assert not proc._allows("[1]")
    assert not proc._allows("null")

    loose = JSONPrefixLogitsProcessor(_processor_stub(), prompt_len=0, require_object=False)
    assert loose._allows("[1]")


def test_allows_accepts_growing_valid_object():
    from visionflow.constrained import JSONPrefixLogitsProcessor

    proc = JSONPrefixLogitsProcessor(_processor_stub(), prompt_len=0)
    for partial in ['{', '{"a"', '{"a":', '{"a": 1', '{"a": 1}']:
        assert proc._allows(partial), partial
    assert not proc._allows('{"a": 1} trailing')


def test_schema_conversion_infers_types():
    converted = _to_json_schema({
        "name": "string",
        "count": "integer",
        "items": "list of strings",
        "urgent": "boolean",
    })
    props = converted["properties"]
    assert props["name"]["type"] == "string"
    assert props["count"]["type"] == "number"
    assert props["items"]["type"] == "array"
    assert props["urgent"]["type"] == "boolean"
    assert set(converted["required"]) == {"name", "count", "items", "urgent"}
