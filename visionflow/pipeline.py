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
    ) -> ExtractionResult:
        """Structured JSON output mode.

        With `constrained=True`, grammar-constrained decoding masks any token that
        would make the output un-completable as JSON, so syntactic validity is
        guaranteed at the decoder rather than patched up afterwards. The repair
        pass is still kept as a fallback for schema-level (not syntax-level)
        failures. See `constrained.py` for the approach and its limits.
        """
        self._ensure_loaded()
        img = self._open_image(image)
        wrapped_prompt = build_json_prompt(prompt, schema=schema)

        processor_factory = None
        if constrained:
            from .constrained import build_json_logits_processor, build_outlines_processor

            outlines_proc = build_outlines_processor(self.engine.model, self.engine.processor, schema)
            if outlines_proc is not None:
                from transformers import LogitsProcessorList

                processor_factory = lambda _prompt_len: LogitsProcessorList([outlines_proc])
            else:
                processor_factory = lambda prompt_len: build_json_logits_processor(
                    self.engine.processor, prompt_len
                )

        raw = self.engine.generate_with_stats(
            img, wrapped_prompt, max_new_tokens=max_new_tokens,
            logits_processor=processor_factory,
        ).text

        def repair_fn(repair_prompt: str) -> str:
            return self.engine.generate(img, repair_prompt, max_new_tokens=max_new_tokens)

        result = extract_json(raw, generate_fn=repair_fn, max_repair_attempts=max_repair_attempts)
        result.constrained = constrained
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
