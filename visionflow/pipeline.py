"""VisionFlow: image + prompt -> structured output, fully local."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

from PIL import Image

from .engine import DEFAULT_MODEL, HardwareTier, ModelEngine
from .extractors import (
    ExtractionResult,
    build_json_prompt,
    build_key_value_prompt,
    extract_json,
    parse_key_value,
)


class VisionFlow:
    """Local, quantized VLM pipeline. Loads once, then call one of the three
    output-mode methods per image.

    Example:
        vf = VisionFlow()
        vf.load()
        vf.text("lab_result.png", "Summarize this document")
        vf.json("lab_result.png", "Extract patient data", schema={"patient_name": "string"})
        vf.key_value("manifest.png", "Extract shipment info", fields=["po_number", "quantity"])
    """

    def __init__(
        self,
        model_id: Optional[str] = None,
        hardware: Optional[HardwareTier] = None,
        force_cpu: bool = False,
    ):
        # model_id=None auto-selects a capability tier for the detected device
        # (SmolVLM-256M on a Pi, 2.25B on an M-series Mac) — see engine.select_model.
        self.engine = ModelEngine(model_id=model_id, hardware=hardware, force_cpu=force_cpu)
        self._loaded = False

    def load(self):
        stats = self.engine.load()
        self._loaded = True
        return stats

    def _ensure_loaded(self):
        if not self._loaded:
            self.load()

    def _open_image(self, image: Union[str, Path, Image.Image]) -> Image.Image:
        if isinstance(image, Image.Image):
            return image.convert("RGB")
        return Image.open(image).convert("RGB")

    def text(
        self,
        image: Union[str, Path, Image.Image],
        prompt: str,
        max_new_tokens: int = 512,
    ) -> str:
        """Free-form text output mode."""
        self._ensure_loaded()
        img = self._open_image(image)
        return self.engine.generate(img, prompt, max_new_tokens=max_new_tokens)

    def json(
        self,
        image: Union[str, Path, Image.Image],
        prompt: str,
        schema: Optional[dict] = None,
        max_new_tokens: int = 768,
        max_repair_attempts: int = 1,
        constrained: bool = False,
        validate: bool = True,
        strict: bool = False,
        speculative: bool = False,
    ) -> ExtractionResult:
        """Structured JSON output mode.

        `constrained=True` masks any token that would make the output
        un-completable as JSON, so syntactic validity is guaranteed at the
        decoder rather than patched up afterwards (see `constrained.py`).

        `validate=True` (the default) additionally checks the parsed object
        against the schema with pydantic — every requested field present, types
        coercible — and feeds any failure back into the repair pass naming the
        offending fields. Constrained decoding guarantees the output parses; this
        is what makes it the object that was actually asked for. `strict=True`
        refuses type coercion, so `"48"` no longer satisfies an integer field.

        `speculative=True` emits the schema-forced parts of the object — the
        keys, colons, commas and braces — without spending a forward pass on
        each, which is possible precisely because the schema pins them. Requires
        a schema; ignored without one. See `speculative.py`, including its note
        on tokenization.

        `schema` may be a `{field: hint}` dict or a pydantic model class.
        """
        self._ensure_loaded()
        img = self._open_image(image)

        from .validation import resolve_model, schema_dict_for_prompt
        from .validation import validate as validate_against

        schema_dict = schema_dict_for_prompt(schema)
        model_cls = resolve_model(schema) if validate else None
        validator = (lambda obj: validate_against(obj, model_cls, strict=strict)) if model_cls else None

        wrapped_prompt = build_json_prompt(prompt, schema=schema_dict)

        processor_factory = None
        if constrained:
            from .constrained import build_json_logits_processor, build_outlines_processor

            outlines_proc = build_outlines_processor(self.engine.model, self.engine.processor, schema_dict)
            if outlines_proc is not None:
                from transformers import LogitsProcessorList

                processor_factory = lambda _prompt_len: LogitsProcessorList([outlines_proc])
            else:
                processor_factory = lambda prompt_len: build_json_logits_processor(
                    self.engine.processor, prompt_len
                )

        speculation_stats = None
        first = None
        if speculative and schema_dict:
            from .speculative import SchemaSkeleton, speculative_generate

            skeleton = SchemaSkeleton.from_schema(schema_dict)
            if skeleton is not None:
                try:
                    _text, first, spec = speculative_generate(
                        self.engine, img, wrapped_prompt, skeleton,
                        max_new_tokens=max_new_tokens,
                    )
                    speculation_stats = spec.as_dict()
                except Exception as e:
                    # Fall back rather than fail: speculation is an optimisation,
                    # and a model whose internals it cannot drive should still
                    # produce an extraction. Recorded so a silent fallback can't
                    # be mistaken for a successful speculative run.
                    speculation_stats = {"error": f"{type(e).__name__}: {e}", "fell_back": True}
                    first = None

        if first is None:
            first = self.engine.generate_with_stats(
                img, wrapped_prompt, max_new_tokens=max_new_tokens,
                logits_processor=processor_factory,
            )

        repair_tally = {"tokens": 0, "passes": 0}

        def repair_fn(repair_prompt: str) -> str:
            stats = self.engine.generate_with_stats(img, repair_prompt, max_new_tokens=max_new_tokens)
            repair_tally["tokens"] += stats.new_tokens
            repair_tally["passes"] += 1
            return stats.text

        result = extract_json(
            first.text, generate_fn=repair_fn,
            max_repair_attempts=max_repair_attempts, validator=validator,
        )
        result.constrained = constrained
        result.new_tokens = first.new_tokens
        result.repair_tokens = repair_tally["tokens"]
        result.forward_passes = 1 + repair_tally["passes"]
        result.speculation = speculation_stats
        return result

    def key_value(
        self,
        image: Union[str, Path, Image.Image],
        prompt: str,
        fields: list[str],
        max_new_tokens: int = 512,
    ) -> dict:
        """Key-value extraction output mode."""
        self._ensure_loaded()
        img = self._open_image(image)
        wrapped_prompt = build_key_value_prompt(prompt, fields)
        raw = self.engine.generate(img, wrapped_prompt, max_new_tokens=max_new_tokens)
        return parse_key_value(raw)
