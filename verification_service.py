from __future__ import annotations

from difflib import SequenceMatcher
from typing import Optional

from models import VerificationField


def _normalize_text(value: Optional[str]) -> str:
    return (value or "").strip().lower()


def similarity(a: Optional[str], b: Optional[str]) -> float:
    a_n = _normalize_text(a)
    b_n = _normalize_text(b)
    if not a_n and not b_n:
        return 1.0
    if not a_n or not b_n:
        return 0.0
    return SequenceMatcher(None, a_n, b_n).ratio()


def compare_field(field: str, a: Optional[str], b: Optional[str], threshold: float = 0.86) -> VerificationField:
    score = similarity(a, b)
    return VerificationField(
        field=field,
        value_a=a,
        value_b=b,
        similarity_score=round(score, 4),
        is_match=score >= threshold,
    )
