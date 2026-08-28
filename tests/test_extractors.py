from visionflow.extractors import (
    build_json_prompt,
    build_key_value_prompt,
    extract_json,
    parse_key_value,
)


def test_extract_json_clean():
    r = extract_json('{"a": 1, "b": "x"}')
    assert r.ok
    assert r.parsed == {"a": 1, "b": "x"}
    assert not r.repaired


def test_extract_json_code_fenced():
    r = extract_json('```json\n{"a": 1}\n```')
    assert r.ok
    assert r.parsed == {"a": 1}


def test_extract_json_with_surrounding_prose():
    r = extract_json('Sure! Here is the data: {"a": 1, "b": 2} Hope that helps.')
    assert r.ok
    assert r.parsed == {"a": 1, "b": 2}


def test_extract_json_repair_pass_fixes_malformed_output():
    def fake_repair(prompt):
        assert "error" not in prompt.lower() or "Text:" in prompt
        return '{"a": 1}'

    r = extract_json("{a: 1,}", generate_fn=fake_repair)
    assert r.ok
    assert r.repaired
    assert r.parsed == {"a": 1}


def test_extract_json_gives_up_without_repair_fn():
    r = extract_json("not json at all", generate_fn=None)
    assert not r.ok
    assert r.error is not None


def test_extract_json_repair_exhausts_attempts():
    def always_broken(prompt):
        return "still not json"

    r = extract_json("also not json", generate_fn=always_broken, max_repair_attempts=2)
    assert not r.ok
    assert r.repaired


def test_parse_key_value():
    result = parse_key_value("po_number: PO-1234\nquantity: 500\nnotes: null\n\nignored line")
    assert result == {"po_number": "PO-1234", "quantity": "500", "notes": None}


def test_build_json_prompt_includes_schema():
    prompt = build_json_prompt("Extract fields", schema={"name": "string"})
    assert "Extract fields" in prompt
    assert '"name"' in prompt


def test_build_json_prompt_without_schema():
    prompt = build_json_prompt("Extract fields")
    assert "Extract fields" in prompt


def test_build_key_value_prompt():
    prompt = build_key_value_prompt("Extract shipment info", ["po_number", "qty"])
    assert "po_number" in prompt and "qty" in prompt
    assert "Extract shipment info" in prompt
