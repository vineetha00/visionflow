"""Grammar-constrained JSON decoding.

The repair loop in `extractors.py` fixes malformed JSON *after* generation: it
costs a second full forward pass and it can still fail. Constraining the decoder
instead makes malformed JSON unrepresentable — at each step, any token that would
make the output impossible to complete as valid JSON has its logit masked to -inf.

Three paths, in order of preference:

1. `outlines`, if installed — a mature regex/CFG-guided decoding library.
2. The built-in `JSONPrefixLogitsProcessor` below — no extra dependency, works
   with any HuggingFace tokenizer.
3. GBNF grammars (`json_schema_to_gbnf`) for the llama.cpp / GGUF CPU path, where
   constraint enforcement happens in C++ rather than in Python.

**Honest limitation of path 2**: validating every token in a ~49k-token vocabulary
at every step is too slow in Python, so the built-in processor validates only the
top-`k` candidates by logit (default 256) and masks everything else. This is an
approximation: it guarantees valid JSON, but it restricts sampling to the top-k
set even when the model's preferred token was valid and outside it. With k=256 and
greedy/low-temperature decoding — the default for extraction — the constrained
argmax matches the unconstrained argmax whenever the latter is grammar-valid, so
in practice this changes output only where the raw model would have produced
invalid JSON. Path 1 and path 3 constrain over the full vocabulary.

This layer guarantees *syntactic* validity only. Whether the extracted values are
correct, and whether they satisfy the requested schema's semantics, is still the
repair loop's and the caller's problem — so `pipeline.json(constrained=True)`
keeps the repair pass as a fallback for schema-level failures.
"""

from __future__ import annotations

import json
from typing import Optional

# ---------------------------------------------------------------------------
# Incremental JSON prefix validator
# ---------------------------------------------------------------------------

_WS = " \t\n\r"


def _scan_string(s: str, i: int) -> Optional[int]:
    """`i` points at the opening quote. Returns the index just past the closing
    quote, -1 if the string is unterminated (a valid prefix), or None if invalid."""
    i += 1
    n = len(s)
    while i < n:
        c = s[i]
        if c == "\\":
            if i + 1 >= n:
                return -1
            esc = s[i + 1]
            if esc in '"\\/bfnrt':
                i += 2
                continue
            if esc == "u":
                hexpart = s[i + 2 : i + 6]
                if len(hexpart) < 4:
                    return -1 if all(ch in "0123456789abcdefABCDEF" for ch in hexpart) else None
                if not all(ch in "0123456789abcdefABCDEF" for ch in hexpart):
                    return None
                i += 6
                continue
            return None
        if c == '"':
            return i + 1
        if c in "\n\r":
            return None  # unescaped control character
        i += 1
    return -1


def _scan_number(s: str, i: int) -> Optional[int]:
    n = len(s)
    start = i
    if i < n and s[i] == "-":
        i += 1
    digits_start = i
    while i < n and s[i].isdigit():
        i += 1
    if i == digits_start:
        return None if i < n else i
    if i < n and s[i] == ".":
        i += 1
        frac_start = i
        while i < n and s[i].isdigit():
            i += 1
        if i == frac_start and i < n:
            return None
    if i < n and s[i] in "eE":
        i += 1
        if i < n and s[i] in "+-":
            i += 1
        exp_start = i
        while i < n and s[i].isdigit():
            i += 1
        if i == exp_start and i < n:
            return None
    return i


_LITERALS = ("true", "false", "null")


def _scan_literal(s: str, i: int) -> Optional[int]:
    for lit in _LITERALS:
        if s.startswith(lit, i):
            return i + len(lit)
    rest = s[i:]
    if any(lit.startswith(rest) for lit in _LITERALS):
        return -1
    return None


def json_prefix_state(s: str) -> Optional[bool]:
    """Classify `s` as a JSON prefix.

    Returns True if `s` is a complete JSON document, False if it is a valid but
    incomplete prefix, and None if no continuation can make it valid JSON.
    """
    i, n = 0, len(s)
    stack: list[str] = []
    state = "value"

    while True:
        while i < n and s[i] in _WS:
            i += 1
        if i >= n:
            if state == "done" or (state == "after_value" and not stack):
                return True
            return False

        c = s[i]

        if state == "value":
            if c == "{":
                stack.append("obj")
                state = "obj_key_or_close"
                i += 1
            elif c == "[":
                stack.append("arr")
                state = "value_or_close"
                i += 1
            elif c == '"':
                r = _scan_string(s, i)
                if r is None:
                    return None
                if r == -1:
                    return False
                i, state = r, "after_value"
            elif c == "-" or c.isdigit():
                r = _scan_number(s, i)
                if r is None:
                    return None
                if r >= n:
                    return False  # more digits could still follow
                i, state = r, "after_value"
            elif c in "tfn":
                r = _scan_literal(s, i)
                if r is None:
                    return None
                if r == -1:
                    return False
                i, state = r, "after_value"
            else:
                return None

        elif state == "value_or_close":
            if c == "]":
                stack.pop()
                i += 1
                state = "after_value"
            else:
                state = "value"

        elif state == "obj_key_or_close":
            if c == "}":
                stack.pop()
                i += 1
                state = "after_value"
            elif c == '"':
                r = _scan_string(s, i)
                if r is None:
                    return None
                if r == -1:
                    return False
                i, state = r, "colon"
            else:
                return None

        elif state == "obj_key":
            if c != '"':
                return None
            r = _scan_string(s, i)
            if r is None:
                return None
            if r == -1:
                return False
            i, state = r, "colon"

        elif state == "colon":
            if c != ":":
                return None
            i += 1
            state = "value"

        elif state == "after_value":
            if not stack:
                state = "done"
                continue
            top = stack[-1]
            if c == ",":
                i += 1
                state = "obj_key" if top == "obj" else "value"
            elif c == "}" and top == "obj":
                stack.pop()
                i += 1
            elif c == "]" and top == "arr":
                stack.pop()
                i += 1
            else:
                return None

        elif state == "done":
            return None


def is_valid_json_prefix(s: str) -> bool:
    return json_prefix_state(s) is not None


def is_complete_json(s: str) -> bool:
    return json_prefix_state(s) is True


# ---------------------------------------------------------------------------
# HuggingFace LogitsProcessor
# ---------------------------------------------------------------------------


def build_json_logits_processor(processor, prompt_len: int, top_k: int = 256):
    """Build a LogitsProcessorList that constrains generation to valid JSON.

    `processor` is the VLM's AutoProcessor (its `.tokenizer` is used for decoding).
    `prompt_len` is the number of prompt tokens to skip when reconstructing the
    generated text so far.
    """
    from transformers import LogitsProcessorList

    return LogitsProcessorList([JSONPrefixLogitsProcessor(processor, prompt_len, top_k=top_k)])


try:  # pragma: no cover - import shape varies across transformers versions
    from transformers import LogitsProcessor as _LogitsProcessorBase
except Exception:  # pragma: no cover
    _LogitsProcessorBase = object


class JSONPrefixLogitsProcessor(_LogitsProcessorBase):
    """Masks any top-k token that would make the output un-completable as JSON.

    See the module docstring for the top-k approximation and its consequences.
    """

    def __init__(self, processor, prompt_len: int, top_k: int = 256):
        self.tokenizer = getattr(processor, "tokenizer", processor)
        self.prompt_len = prompt_len
        self.top_k = top_k
        self._decode_cache: dict[int, str] = {}
        self.eos_token_id = getattr(self.tokenizer, "eos_token_id", None)
        # Diagnostics — surfaced in benchmarks so the cost of constraining is visible.
        self.steps = 0
        self.tokens_masked = 0
        self.fallback_steps = 0

    def _token_text(self, token_id: int) -> str:
        text = self._decode_cache.get(token_id)
        if text is None:
            text = self.tokenizer.decode([token_id], skip_special_tokens=True)
            self._decode_cache[token_id] = text
        return text

    def __call__(self, input_ids, scores):
        import torch

        self.steps += 1
        generated_ids = input_ids[0, self.prompt_len :]
        prefix = self.tokenizer.decode(generated_ids, skip_special_tokens=True)

        # Leading whitespace/prose before the first '{' is tolerated by the
        # extractor's brace-scan, so only start constraining once a '{' appears.
        brace = prefix.find("{")
        if brace == -1:
            return scores
        json_prefix = prefix[brace:]

        k = min(self.top_k, scores.shape[-1])
        top_scores, top_indices = torch.topk(scores[0], k)

        mask = torch.full_like(scores, float("-inf"))
        allowed_any = False
        complete = is_complete_json(json_prefix)

        for score, idx in zip(top_scores.tolist(), top_indices.tolist()):
            if self.eos_token_id is not None and idx == self.eos_token_id:
                # EOS is only legal once the object actually closes.
                if complete:
                    mask[0, idx] = score
                    allowed_any = True
                continue
            piece = self._token_text(idx)
            if piece == "":
                continue
            if is_valid_json_prefix(json_prefix + piece):
                mask[0, idx] = score
                allowed_any = True
            else:
                self.tokens_masked += 1

        if not allowed_any:
            # No valid continuation in the top-k window. Rather than emit invalid
            # JSON, force EOS if the object is complete, otherwise fall back to the
            # unconstrained distribution and let the repair loop handle it — a
            # wrong answer beats a hang.
            self.fallback_steps += 1
            if complete and self.eos_token_id is not None:
                mask[0, self.eos_token_id] = 0.0
                return mask
            return scores

        return mask

    @property
    def diagnostics(self) -> dict:
        return {
            "steps": self.steps,
            "tokens_masked": self.tokens_masked,
            "fallback_steps": self.fallback_steps,
            "top_k": self.top_k,
        }


# ---------------------------------------------------------------------------
# GBNF grammar generation (llama.cpp / GGUF path)
# ---------------------------------------------------------------------------

_GBNF_PREAMBLE = r"""
ws      ::= [ \t\n]*
string  ::= "\"" char* "\"" ws
char    ::= [^"\\] | "\\" (["\\/bfnrt] | "u" [0-9a-fA-F]{4})
number  ::= "-"? ([0-9] | [1-9] [0-9]*) ("." [0-9]+)? ([eE] [-+]? [0-9]+)? ws
boolean ::= ("true" | "false") ws
null    ::= "null" ws
value   ::= object | array | string | number | boolean | null
object  ::= "{" ws (string ":" ws value ("," ws string ":" ws value)*)? "}" ws
array   ::= "[" ws (value ("," ws value)*)? "]" ws
""".strip()


def json_schema_to_gbnf(schema: Optional[dict] = None) -> str:
    """Emit a GBNF grammar for llama.cpp.

    With a schema, the root rule pins the exact key sequence, so the model cannot
    invent, drop, or reorder fields. Without one, the root is any JSON object.
    """
    if not schema:
        return f"root ::= object ws\n{_GBNF_PREAMBLE}\n"

    parts = []
    for key in schema:
        escaped = key.replace("\\", "\\\\").replace('"', '\\"')
        parts.append(f'"\\"{escaped}\\"" ws ":" ws value')
    root = 'root ::= "{" ws ' + ' "," ws '.join(parts) + ' "}" ws'
    return f"{root}\n{_GBNF_PREAMBLE}\n"


# ---------------------------------------------------------------------------
# outlines integration (optional)
# ---------------------------------------------------------------------------


def outlines_available() -> bool:
    try:
        import outlines  # noqa: F401

        return True
    except Exception:
        return False


def build_outlines_processor(model, processor, schema: Optional[dict] = None):
    """Build an outlines-backed logits processor, or None if outlines isn't installed.

    Unlike the built-in processor this constrains over the full vocabulary via a
    precompiled FSM, so it has no top-k approximation.
    """
    if not outlines_available():
        return None
    try:
        from outlines.processors import JSONLogitsProcessor
        from outlines.models.transformers import TransformerTokenizer

        tokenizer = TransformerTokenizer(getattr(processor, "tokenizer", processor))
        json_schema = _to_json_schema(schema) if schema else {"type": "object"}
        return JSONLogitsProcessor(json_schema, tokenizer)
    except Exception:
        return None


def _to_json_schema(schema: dict) -> dict:
    """Convert VisionFlow's loose {field: "description"} schema into JSON Schema."""
    properties = {}
    for key, hint in schema.items():
        hint_l = str(hint).lower()
        if "list" in hint_l or "array" in hint_l:
            properties[key] = {"type": "array", "items": {"type": "string"}}
        elif "int" in hint_l or "number" in hint_l or "float" in hint_l:
            properties[key] = {"type": "number"}
        elif "bool" in hint_l:
            properties[key] = {"type": "boolean"}
        else:
            properties[key] = {"type": "string"}
    return {
        "type": "object",
        "properties": properties,
        "required": list(schema.keys()),
    }
