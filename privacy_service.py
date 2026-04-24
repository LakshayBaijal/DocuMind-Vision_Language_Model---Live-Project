from __future__ import annotations

import base64
from typing import Any, Dict, List, Optional

import cv2
import numpy as np


def _to_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _safe_bbox(bbox: List[float], width: int, height: int) -> tuple[int, int, int, int] | None:
    if len(bbox) != 4:
        return None
    x1, y1, x2, y2 = (_to_float(v) for v in bbox)
    if None in (x1, y1, x2, y2):
        return None
    x1 = max(0, min(width - 1, int(x1 * width)))
    y1 = max(0, min(height - 1, int(y1 * height)))
    x2 = max(0, min(width, int(x2 * width)))
    y2 = max(0, min(height, int(y2 * height)))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


class PrivacyService:
    @staticmethod
    def redact_pii_regions(image_bytes: bytes, pii_regions: List[Dict[str, Any]]) -> str:
        """
        Redact sensitive regions using heavy Gaussian blur.
        Returns redacted image as base64-encoded JPEG.
        """
        np_arr = np.frombuffer(image_bytes, dtype=np.uint8)
        image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if image is None:
            return ""

        height, width = image.shape[:2]
        out = image.copy()
        for region in pii_regions or []:
            bbox = _safe_bbox(region.get("bbox") or [], width, height)
            if not bbox:
                continue
            x1, y1, x2, y2 = bbox
            roi = out[y1:y2, x1:x2]
            if roi.size == 0:
                continue
            blur = cv2.GaussianBlur(roi, (51, 51), sigmaX=24, sigmaY=24)
            out[y1:y2, x1:x2] = blur

        ok, encoded = cv2.imencode(".jpg", out, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        if not ok:
            return ""
        return base64.b64encode(encoded.tobytes()).decode("ascii")
