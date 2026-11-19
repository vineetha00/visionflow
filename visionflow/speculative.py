"""Schema-structural speculative decoding for JSON extraction.

When the schema pins the field order, most of the output is not a prediction at
all. For `{"mrn": "string", "qty": "integer"}` the model only has to produce two
values; everything else — `{"mrn": `, then `, "qty": `, then `}` — is forced by
the schema. A plain decode loop spends one forward pass per token on all of it.

This module emits those forced runs directly and lets the *next* forward pass
absorb the whole run at once, which is what speculative decoding's verification
step does. The difference from draft-model speculation is that there is nothing
to verify: the forced text is the only continuation the grammar permits, so the
acceptance rate is 100% by construction, there is no second model in memory, and
there is no rejection path to get wrong.

The saving is real but bounded by the shape of the schema. A 10-field medical
record whose values are short identifiers is mostly skeleton; one field holding
a paragraph of findings is mostly value. `SpeculationStats.tokens_saved` reports
what was actually skipped per call rather than leaving it to be assumed — this
project's benchmark harness exists because that distinction matters.

**Honest caveat on tokenization.** Forced text is tokenized on its own and
spliced in, so the token boundaries may differ from what the model would have
produced for the same characters in context. The decoded string is identical
either way and the schema pins what follows, but the KV cache then holds a
slightly unusual segmentation, which can nudge the distribution over the
following *value* tokens. Measurable as a difference in extraction accuracy
between speculative and non-speculative runs; `vf accuracy --speculative`
compares them directly instead of assuming it is negligible.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from .constrained import _scan_literal, _scan_number, _scan_string

_WS = " \t\n\r"


def _skip_ws(s: str, i: int) -> int:
    while i < len(s) and s[i] in _WS:
        i += 1
    return i


def _scan_value(s: str, i: int) -> Optional[int]:
    """Scan one JSON value starting at `i`.

    Returns the index just past it, -1 if it is a valid but incomplete prefix,
    or None if it cannot be a JSON value.
    """
    if i >= len(s):
        return -1
    c = s[i]
    if c == '"':
        return _scan_string(s, i)
    if c == "-" or c.isdigit():
        end = _scan_number(s, i)
        if end is None:
            return None
        # A number running to the end of the buffer may still be growing.
        return -1 if end >= len(s) else end
    if c in "tfn":
        return _scan_literal(s, i)
    if c in "[{":
        return _scan_container(s, i)
    return None


def _scan_container(s: str, i: int) -> Optional[int]:
    """Balanced scan of an array or object, respecting strings and escapes."""
    opener = s[i]
    closer = "]" if opener == "[" else "}"
    depth = 0
    while i < len(s):
        c = s[i]
        if c == '"':
            end = _scan_string(s, i)
            if end is None:
                return None
            if end == -1:
                return -1
            i = end
            continue
        if c in "[{":
            depth += 1
        elif c in "]}":
            depth -= 1
            if depth == 0:
                return i + 1 if c == closer else None
            if depth < 0:
                return None
        i += 1
    return -1


@dataclass
class SpeculationStats:
    """What the speculator actually did, per generation."""

    forced_runs: int = 0
    forced_tokens: int = 0
    forward_passes: int = 0
    free_tokens: int = 0
    diverged: bool = False

    @property
    def tokens_saved(self) -> int:
        """Forward passes avoided: a forced run of k tokens costs one pass, not k."""
        return max(0, self.forced_tokens - self.forced_runs)

    @property
    def total_tokens(self) -> int:
        return self.forced_tokens + self.free_tokens

    def as_dict(self) -> dict:
        return {
            "forced_runs": self.forced_runs,
            "forced_tokens": self.forced_tokens,
            "free_tokens": self.free_tokens,
            "forward_passes": self.forward_passes,
            "tokens_saved": self.tokens_saved,
            "diverged": self.diverged,
        }


class SchemaSkeleton:
    """The deterministic parts of a JSON object with a pinned field order.

    `forced_continuation(prefix)` answers the only question the decode loop asks:
    given what has been emitted so far, is the next text forced by the schema, or
    does the model have to produce a value?

    Returns the forced string, `""` when a free value comes next, and `None` when
    the prefix has diverged from the skeleton — which is the caller's signal to
    stop speculating and finish with ordinary decoding rather than force the
    output into a shape it has already left.
    """

    def __init__(self, keys: list[str], types: Optional[list] = None):
        self.keys = list(keys)
        # Field types let the skeleton force the opening quote of a string value.
        # That is not just one more token saved: without it the model is free to
        # open an unquoted value, which is how a live run produced
        # `{"manifest_no": 2026-` — a date the model clearly read correctly but
        # emitted as bare characters, diverging from the skeleton and from JSON.
        # Forcing the quote makes that whole failure mode unrepresentable.
        self.types = list(types) if types else [str] * len(self.keys)

    @classmethod
    def from_schema(cls, schema) -> Optional["SchemaSkeleton"]:
        from .validation import python_type_for_hint, schema_dict_for_prompt

        as_dict = schema_dict_for_prompt(schema)
        if not as_dict:
            return None
        return cls(
            list(as_dict.keys()),
            [python_type_for_hint(h) for h in as_dict.values()],
        )

    def _key_literal(self, index: int) -> str:
        import json as _json

        return _json.dumps(self.keys[index])

    def forced_continuation(self, prefix: str) -> Optional[str]:
        if not self.keys:
            return None

        s = prefix
        i = _skip_ws(s, 0)

        if i >= len(s):
            return "{" + self._key_literal(0) + ": "
        if s[i] != "{":
            return None
        i += 1

        for index, _key in enumerate(self.keys):
            expected_key = self._key_literal(index)

            i = _skip_ws(s, i)
            if i >= len(s):
                return expected_key + ": "
            # The key may be only partly emitted, in which case the remainder of
            # it is forced too.
            rest = s[i:]
            if not s.startswith(expected_key, i):
                if expected_key.startswith(rest):
                    return expected_key[len(rest):] + ": "
                return None
            i += len(expected_key)

            i = _skip_ws(s, i)
            if i >= len(s):
                return ": "
            if s[i] != ":":
                return None
            i += 1

            i = _skip_ws(s, i)
            if i >= len(s):
                # The value comes next. For a string field its opening quote is
                # forced too — the model only chooses the contents.
                return '"' if self.types[index] is str else ""
            end = _scan_value(s, i)
            if end is None:
                return None
            if end == -1:
                return ""  # mid-value
            i = end

            i = _skip_ws(s, i)
            last = index == len(self.keys) - 1
            if i >= len(s):
                return "}" if last else ", " + self._key_literal(index + 1) + ": "
            if last:
                return None if s[i] != "}" else ""
            if s[i] != ",":
                return None
            i += 1

        return ""

    def is_complete(self, prefix: str) -> bool:
        """Has the full skeleton been emitted, with every value filled?"""
        s = prefix.rstrip()
        if not s.endswith("}"):
            return False
        return self.forced_continuation(prefix) == "" and self._all_values_present(s)

    def _all_values_present(self, s: str) -> bool:
        import json as _json

        try:
            parsed = _json.loads(s)
        except Exception:
            return False
        return isinstance(parsed, dict) and all(k in parsed for k in self.keys)


def speculative_generate(
    engine,
    image,
    prompt: str,
    skeleton: SchemaSkeleton,
    max_new_tokens: int = 512,
):
    """Greedy decode with the schema's forced spans spliced in.

    Returns `(text, GenerationStats, SpeculationStats)`. On any failure this
    raises; the caller is expected to fall back to ordinary generation rather
    than silently returning a worse result under the same name.
    """
    import time

    import torch

    from .engine import GenerationStats, HardwareTier

    model, processor = engine.model, engine.processor
    if model is None:
        raise RuntimeError("Call .load() before generating")
    tokenizer = getattr(processor, "tokenizer", processor)

    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
    text_prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(text=text_prompt, images=[image], return_tensors="pt")
    device = engine.hardware.value if engine.hardware != HardwareTier.CPU else "cpu"
    inputs = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in inputs.items()}

    eos_id = getattr(tokenizer, "eos_token_id", None)
    stats = SpeculationStats()
    generated: list[int] = []

    start = time.perf_counter()
    with torch.no_grad():
        out = model(**inputs, use_cache=True)
        stats.forward_passes += 1
        past = out.past_key_values
        next_logits = out.logits[:, -1, :]
        attention_mask = inputs.get("attention_mask")

        def advance(token_ids: list[int]):
            """Feed `token_ids` through the model in one pass, refreshing the cache."""
            nonlocal past, next_logits, attention_mask
            ids = torch.tensor([token_ids], device=device, dtype=torch.long)
            kwargs = {"input_ids": ids, "past_key_values": past, "use_cache": True}
            if attention_mask is not None:
                attention_mask = torch.cat(
                    [attention_mask, torch.ones((1, len(token_ids)), device=device,
                                                dtype=attention_mask.dtype)],
                    dim=1,
                )
                kwargs["attention_mask"] = attention_mask
            step = model(**kwargs)
            past = step.past_key_values
            next_logits = step.logits[:, -1, :]
            stats.forward_passes += 1

        while len(generated) < max_new_tokens:
            text = tokenizer.decode(generated, skip_special_tokens=True) if generated else ""
            forced = skeleton.forced_continuation(text)

            if forced is None:
                # The output has left the skeleton. Stop speculating; the caller
                # decides what to do rather than this loop forcing a shape the
                # model has already contradicted.
                stats.diverged = True
                break

            if forced:
                ids = tokenizer(forced, add_special_tokens=False)["input_ids"]
                ids = [i for i in ids if i is not None]
                if not ids:
                    stats.diverged = True
                    break
                remaining = max_new_tokens - len(generated)
                ids = ids[:remaining]
                generated.extend(ids)
                stats.forced_runs += 1
                stats.forced_tokens += len(ids)
                if skeleton.is_complete(tokenizer.decode(generated, skip_special_tokens=True)):
                    break
                advance(ids)
                continue

            if skeleton.is_complete(text):
                break

            token = int(torch.argmax(next_logits, dim=-1).item())
            if eos_id is not None and token == eos_id:
                break
            generated.append(token)
            stats.free_tokens += 1
            advance([token])

    engine._synchronize()
    elapsed = time.perf_counter() - start
    text = tokenizer.decode(generated, skip_special_tokens=True).strip()
    return text, GenerationStats(text=text, new_tokens=len(generated), seconds=elapsed), stats
