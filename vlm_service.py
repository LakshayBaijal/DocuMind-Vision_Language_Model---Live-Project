"""Vision-language extraction via Groq Cloud (no local OCR / object detection)."""

from __future__ import annotations

import base64
import io
import json
import re
from typing import Any, Dict, Optional, Tuple

from groq import Groq
from PIL import Image

from config import Settings, get_settings

VISION_PROMPT = (
    "You are an enterprise document intelligence extractor for KYC and compliance workflows. "
    "Read the image carefully and extract ALL visible details including text, numbers, handwritten entries, "
    "issuer details, identifiers, dates, grades/ranks/scores, addresses, and references. "
    "Return one JSON object only (no markdown/prose) with these exact keys:\n"
    "{"
    '"document_type": string, '
    '"name": string|null, '
    '"id_number": string|null, '
    '"date_of_birth": string|null, '
    '"full_text": string|null, '
    '"summary": string|null, '
    '"key_value_pairs": [{"key": string, "value": string}], '
    '"entities": [{"label": string, "value": string}], '
    '"normalized_data": object, '
    '"field_audit": [{"field": string, "value": string|null, "confidence_score": number 0..1, '
    '"reasoning": string, "is_critical": boolean}], '
    '"pii_regions": [{"label": string, "value": string|null, "bbox": [x1,y1,x2,y2], '
    '"confidence_score": number 0..1}]'
    "}\n"
    "For pii_regions, bbox coordinates MUST be normalized between 0 and 1 where [0,0] is top-left and [1,1] is "
    "bottom-right. Include regions for PAN/Aadhaar/ID numbers, personal addresses, and date of birth if visible."
)

NORMALIZATION_PROMPT = (
    "You are a strict JSON normalizer. You will receive extracted document JSON. "
    "Keep facts grounded only in provided data, do not hallucinate. "
    "Return one JSON object with the exact same top-level keys: "
    "document_type, name, id_number, date_of_birth, full_text, summary, key_value_pairs, entities, normalized_data, "
    "field_audit, pii_regions. "
    "Ensure key_value_pairs is an array of {key,value} and entities is an array of {label,value}. "
    "Ensure field_audit is an array of {field,value,confidence_score,reasoning,is_critical}. "
    "Ensure pii_regions is an array of {label,value,bbox,confidence_score}. "
    "In normalized_data include any useful derived structure such as: "
    "certificate_title, issuing_organization, class_or_grade_level, session, score_or_grade, position_or_rank, "
    "exam_name, subject_area, and other relevant fields when present."
)


def _strip_json_fences(text: str) -> str:
    t = text.strip()
    if t.startswith("```"):
        t = re.sub(r"^```(?:json)?\s*", "", t, flags=re.IGNORECASE)
        t = re.sub(r"\s*```$", "", t)
    return t.strip()


def _encode_image_data_url(image_bytes: bytes, mime: str, settings: Settings) -> Tuple[str, str]:
    """Return (data_url, effective_mime) keeping encoded image under Groq request size limits."""
    max_bytes = settings.groq_max_image_bytes
    mime = (mime or "image/jpeg").split(";")[0].strip().lower()
    if mime not in ("image/jpeg", "image/png", "image/webp", "image/gif"):
        mime = "image/jpeg"

    if len(image_bytes) <= max_bytes and mime in ("image/jpeg", "image/png"):
        b64 = base64.b64encode(image_bytes).decode("ascii")
        return f"data:{mime};base64,{b64}", mime

    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    scale = 1.0
    for _ in range(14):
        w, h = img.size
        buf = io.BytesIO()
        resized = img if scale >= 0.999 else img.resize(
            (max(1, int(w * scale)), max(1, int(h * scale))),
            Image.Resampling.LANCZOS,
        )
        resized.save(buf, format="JPEG", quality=82, optimize=True)
        data = buf.getvalue()
        if len(data) <= max_bytes:
            b64 = base64.b64encode(data).decode("ascii")
            return "data:image/jpeg;base64," + b64, "image/jpeg"
        scale *= 0.82

    b64 = base64.b64encode(data).decode("ascii")
    return "data:image/jpeg;base64," + b64, "image/jpeg"


class VLMService:
    """Singleton-style Groq vision client for document extraction."""

    _instance: Optional["VLMService"] = None

    def __init__(self, settings: Optional[Settings] = None) -> None:
        self._settings = settings or get_settings()
        self._client = Groq(api_key=self._settings.groq_api_key)

    @classmethod
    def instance(cls) -> "VLMService":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def extract_document_data(self, image_bytes: bytes, mime_type: Optional[str] = None) -> str:
        data_url, _ = _encode_image_data_url(image_bytes, mime_type or "image/jpeg", self._settings)
        kwargs: Dict[str, Any] = dict(
            model=self._settings.groq_vision_model,
            temperature=self._settings.llm_temperature,
            max_completion_tokens=self._settings.llm_max_tokens,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": VISION_PROMPT},
                        {
                            "type": "image_url",
                            "image_url": {"url": data_url},
                        },
                    ],
                }
            ],
        )
        if self._settings.groq_response_json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        completion = self._client.chat.completions.create(**kwargs)
        content = completion.choices[0].message.content
        if not content:
            raise RuntimeError("Groq returned empty content")
        first_pass = content.strip()

        # Second pass: text-only LLM normalization for richer structure.
        normalized_kwargs: Dict[str, Any] = dict(
            model=self._settings.llm_model,
            temperature=0,
            max_completion_tokens=self._settings.llm_max_tokens,
            messages=[
                {"role": "system", "content": NORMALIZATION_PROMPT},
                {"role": "user", "content": first_pass},
            ],
        )
        if self._settings.groq_response_json_mode:
            normalized_kwargs["response_format"] = {"type": "json_object"}

        normalized = self._client.chat.completions.create(**normalized_kwargs)
        normalized_content = normalized.choices[0].message.content
        if normalized_content:
            return normalized_content.strip()
        return first_pass


def parse_document_json(raw: str) -> Tuple[Dict[str, Any], str]:
    """Return (parsed dict, cleaned json string). Raises json.JSONDecodeError on failure."""
    cleaned = _strip_json_fences(raw)
    parsed = json.loads(cleaned)
    if not isinstance(parsed, dict):
        raise json.JSONDecodeError("Expected JSON object", cleaned, 0)
    return parsed, cleaned
