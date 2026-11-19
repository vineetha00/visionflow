"""Schema validation for extracted JSON, via pydantic.

Constrained decoding (`constrained.py`) guarantees the output *parses*. It says
nothing about whether the object is the one that was asked for: whether every
requested field came back, whether `qty` is a number rather than the word
"forty-eight", whether `flagged_anomalies` is a list. That is this module's job,
and it is the difference between "parseable" and "usable downstream".

It also sharpens the repair loop. Without validation, a repair prompt can only
say what the JSON parser said -- `Expecting ',' delimiter: line 1 column 92`,
which tells the model nothing about the task. With validation it can say
`qty: input should be a valid number; mrn: field required`, which names the
field and the problem. The second is a far better instruction.

Two design decisions worth stating, because both are deliberate:

**Fields are required but nullable.** The JSON-mode prompt instructs the model to
emit `null` for anything absent from the image rather than omitting the key, and
the accuracy harness counts an object as schema-valid only when every requested
key is present. So a missing key is a real failure the repair loop should see,
while an explicit `null` is a correct answer. In pydantic terms that is
`Optional[T]` with no default: present-and-null passes, absent fails.

**Coercion is lenient by default.** A VLM reading a scanned document returns
`"48"` for a quantity far more often than `48`, and rejecting that would make
validation an obstacle rather than a safety net. Pydantic's non-strict mode
coerces it. Pass `strict=True` when the distinction matters -- it is a real
signal that the model is guessing at types rather than reading them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Union

# Loose type hints used in VisionFlow schemas (`{"qty": "integer"}`) mapped to
# Python types. Matched by substring against a lowercased hint, so "string
# (YYYY-MM-DD)" and "list of strings" both resolve.
_HINT_TYPES: list[tuple[tuple[str, ...], Any]] = [
    (("list", "array"), list),
    (("bool",), bool),
    (("int",), int),
    (("number", "float", "decimal"), float),
    (("string", "str", "text"), str),
]


def python_type_for_hint(hint: Any) -> Any:
    """Map a loose schema hint to a Python type, defaulting to `str`.

    `str` is the default on purpose: document extraction is textual, and a hint
    this function does not recognise is far more likely to be a prose
    description of a string field than a number.
    """
    if isinstance(hint, type):
        return hint
    text = str(hint).lower()
    for needles, py_type in _HINT_TYPES:
        if any(n in text for n in needles):
            return py_type
    return str


def is_pydantic_model(obj: Any) -> bool:
    try:
        from pydantic import BaseModel

        return isinstance(obj, type) and issubclass(obj, BaseModel)
    except Exception:
        return False


def model_from_schema(schema: dict, name: str = "ExtractionSchema"):
    """Build a pydantic model from a VisionFlow `{field: hint}` schema.

    Every field is `Optional[T]` and required: an explicit null is accepted, a
    missing key is not. See the module docstring for why.
    """
    from pydantic import create_model

    fields: dict[str, Any] = {}
    for key, hint in schema.items():
        py_type = python_type_for_hint(hint)
        # `...` is pydantic's "required" sentinel. Optional[T] without a default
        # means the key must appear but may be null.
        fields[key] = (Optional[py_type], ...)
    return create_model(name, **fields)


def resolve_model(schema: Union[dict, Any, None]):
    """Accept either a pydantic model class or a loose dict schema. None disables."""
    if schema is None:
        return None
    if is_pydantic_model(schema):
        return schema
    if isinstance(schema, dict):
        return model_from_schema(schema)
    return None


def schema_dict_for_prompt(schema: Union[dict, Any, None]) -> Optional[dict]:
    """Render a schema as the `{field: hint}` dict the prompt builder expects.

    A pydantic model passed by the caller is flattened back into hints so the
    prompt, the GBNF grammar and the speculative skeleton all see the same field
    order as the validator.
    """
    if schema is None:
        return None
    if isinstance(schema, dict):
        return schema
    if is_pydantic_model(schema):
        out: dict[str, str] = {}
        for key, info in schema.model_fields.items():
            annotation = getattr(info, "annotation", str)
            out[key] = _describe_annotation(annotation)
        return out
    return None


def _describe_annotation(annotation: Any) -> str:
    """A short, human- and model-readable description of a pydantic annotation."""
    import typing

    origin = typing.get_origin(annotation)
    if origin is Union:
        inner = [a for a in typing.get_args(annotation) if a is not type(None)]
        if inner:
            return _describe_annotation(inner[0])
    if origin in (list, set, tuple):
        args = typing.get_args(annotation)
        if args:
            return f"list of {_describe_annotation(args[0])}s"
        return "list"
    if annotation is bool:
        return "boolean"
    if annotation is int:
        return "integer"
    if annotation is float:
        return "number"
    if annotation is str:
        return "string"
    return getattr(annotation, "__name__", "string")


@dataclass
class ValidationOutcome:
    ok: bool
    validated: Optional[dict] = None
    errors: list[dict] = field(default_factory=list)

    @property
    def summary(self) -> str:
        """Compact, field-oriented error text suitable for a repair prompt.

        Deliberately not pydantic's full error rendering, which carries URLs and
        input echoes that waste tokens the repair pass has to pay for.
        """
        if self.ok:
            return ""
        parts = []
        for err in self.errors:
            loc = ".".join(str(p) for p in err.get("loc", ())) or "(root)"
            parts.append(f"{loc}: {err.get('msg', 'invalid')}")
        return "; ".join(parts)

    @property
    def missing_fields(self) -> list[str]:
        return [
            ".".join(str(p) for p in e.get("loc", ()))
            for e in self.errors
            if str(e.get("type", "")).startswith("missing")
        ]


def validate(parsed: Any, model, strict: bool = False) -> ValidationOutcome:
    """Validate `parsed` against a pydantic model.

    Returns an outcome rather than raising: a schema failure is an expected part
    of extraction, handled by the repair loop, not an exceptional condition.
    """
    if model is None:
        return ValidationOutcome(ok=True, validated=parsed if isinstance(parsed, dict) else None)
    if not isinstance(parsed, dict):
        return ValidationOutcome(
            ok=False,
            errors=[{"loc": ("(root)",), "msg": f"expected a JSON object, got {type(parsed).__name__}",
                     "type": "object_type"}],
        )

    from pydantic import ValidationError

    try:
        instance = model.model_validate(parsed, strict=strict)
    except ValidationError as exc:
        return ValidationOutcome(ok=False, errors=list(exc.errors()))
    return ValidationOutcome(ok=True, validated=instance.model_dump())
