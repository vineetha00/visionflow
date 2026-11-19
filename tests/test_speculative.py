"""Tests for schema-structural speculation.

The decode loop needs a model, but the part that decides *what* to skip is pure
logic and this is where it would go wrong. Two failure modes matter and have
opposite consequences:

  - forcing too little wastes the optimisation (harmless, invisible);
  - forcing too much writes tokens the model never chose, corrupting the output
    while still producing valid-looking JSON (harmful, and invisible too).

So the interesting cases are the boundaries: partially-emitted keys, values that
are still growing, and prefixes that have left the skeleton entirely.
"""

import pytest

from visionflow.speculative import SchemaSkeleton, SpeculationStats, _scan_value


@pytest.fixture
def skeleton():
    return SchemaSkeleton(["mrn", "qty"], [str, int])


@pytest.mark.parametrize(
    "prefix,expected",
    [
        ("", '{"mrn": '),                       # nothing yet: open and name field 0
        ("{", '"mrn": '),
        ('{"mr', 'n": '),                       # key half-emitted: finish it
        ('{"mrn"', ": "),
        ('{"mrn":', '"'),                       # string field: its quote is forced too
        ('{"mrn": ', '"'),
        ('{"mrn": "SYN-1', ""),                 # inside the string: model's choice
        ('{"mrn": "SYN-1"', ', "qty": '),       # value closed: bridge to field 1
        ('{"mrn": null', ', "qty": '),          # null is a complete value
        ('{"mrn": "A", "qty": ', ""),           # numeric field: no quote forced
        ('{"mrn": "A", "qty": 48', ""),         # trailing number may still grow
        ('{"mrn": "A", "qty": 48}', ""),        # finished
    ],
)
def test_forced_continuation(skeleton, prefix, expected):
    assert skeleton.forced_continuation(prefix) == expected


def test_string_fields_force_their_opening_quote(skeleton):
    """Regression from a live run: without this the model opened an unquoted
    value (`{"manifest_no": 2026-`), which is neither valid JSON nor on the
    skeleton, so speculation diverged on a field it had read correctly."""
    assert skeleton.forced_continuation('{"mrn": ') == '"'
    # A non-string field must NOT get a quote, or a number becomes a string.
    assert skeleton.forced_continuation('{"mrn": "A", "qty": ') == ""


@pytest.mark.parametrize(
    "prefix",
    [
        '{"other": 1',          # wrong key
        "garbage",              # not an object
        '[1, 2]',               # array at root
        '{"mrn" 1',             # missing colon
        '{"mrn": "A" "qty"',    # missing comma
    ],
)
def test_divergence_returns_none_rather_than_forcing(skeleton, prefix):
    """None is the signal to stop speculating. Forcing a shape the output has
    already contradicted would overwrite tokens the model actually chose."""
    assert skeleton.forced_continuation(prefix) is None


def test_whitespace_between_tokens_is_tolerated(skeleton):
    assert skeleton.forced_continuation('{ "mrn" : "A" , ') == '"qty": '
    assert skeleton.forced_continuation('{\n  "mrn": "A",\n  ') == '"qty": '


def test_keys_needing_escapes_are_emitted_as_valid_json():
    quoted = SchemaSkeleton(['we"ird'])
    forced = quoted.forced_continuation("")
    assert forced.startswith('{"we\\"ird"')


def test_container_values_are_scanned_whole():
    sk = SchemaSkeleton(["flags", "n"])
    assert sk.forced_continuation('{"flags": ["a", "b"]') == ', "n": '
    assert sk.forced_continuation('{"flags": ["a", ') == ""       # still open
    assert sk.forced_continuation('{"flags": ["a{]", "b"]') == ', "n": '  # braces in string


def test_string_values_containing_delimiters_do_not_end_the_value_early():
    sk = SchemaSkeleton(["note", "n"])
    assert sk.forced_continuation('{"note": "a, b: c}"') == ', "n": '
    assert sk.forced_continuation('{"note": "escaped \\" quote"') == ', "n": '


def test_is_complete_requires_every_field(skeleton):
    assert skeleton.is_complete('{"mrn": "A", "qty": 1}')
    assert not skeleton.is_complete('{"mrn": "A"}')
    assert not skeleton.is_complete('{"mrn": "A", "qty": 1')


def test_empty_schema_speculates_nothing():
    assert SchemaSkeleton([]).forced_continuation("") is None
    assert SchemaSkeleton.from_schema({}) is None


def test_from_schema_accepts_a_pydantic_model():
    from pydantic import BaseModel

    class S(BaseModel):
        a: str
        b: int

    sk = SchemaSkeleton.from_schema(S)
    assert sk.keys == ["a", "b"]


@pytest.mark.parametrize(
    "text,expected",
    [
        ('"done"', 6), ('"open', -1), ("null", 4), ("tru", -1),
        ("[1,2]", 5), ("[1,", -1), ("{}", 2), ("nope", None),
    ],
)
def test_scan_value_boundaries(text, expected):
    assert _scan_value(text, 0) == expected


# --- accounting ------------------------------------------------------------


def test_tokens_saved_counts_passes_avoided_not_tokens_emitted():
    """A forced run of k tokens costs one forward pass, so it saves k-1 — not k.
    Overstating this would inflate the headline number the README reports."""
    stats = SpeculationStats(forced_runs=3, forced_tokens=12, free_tokens=20)
    assert stats.tokens_saved == 9
    assert stats.total_tokens == 32


def test_no_speculation_saves_nothing():
    assert SpeculationStats().tokens_saved == 0
    assert SpeculationStats(forced_runs=4, forced_tokens=4).tokens_saved == 0
