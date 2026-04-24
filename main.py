from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple
import time

from fastapi import Depends, FastAPI, File, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from config import get_settings
from models import DocumentData, ExtractErrorResponse, PairVerificationResponse
from privacy_service import PrivacyService
from verification_service import compare_field
from vlm_service import VLMService, parse_document_json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Visual Document Extractor", version="1.0.0")
_DEBUG_LOG_PATH = Path(__file__).resolve().parent / "debug-a9b039.log"


def _dbg(run_id: str, hypothesis_id: str, location: str, message: str, data: Dict[str, Any]) -> None:
    try:
        payload = {
            "sessionId": "a9b039",
            "runId": run_id,
            "hypothesisId": hypothesis_id,
            "location": location,
            "message": message,
            "data": data,
            "timestamp": int(time.time() * 1000),
        }
        with _DEBUG_LOG_PATH.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=True) + "\n")
    except Exception:
        pass

_settings = get_settings()
_cors_list: List[str] = [
    o.strip() for o in _settings.cors_origins.split(",") if o.strip()
]
if _cors_list:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=_cors_list,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
else:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )


def get_vlm() -> VLMService:
    return VLMService.instance()


def _extract_or_error(vlm: VLMService, image_bytes: bytes, mime_type: str) -> Tuple[Dict[str, Any], str]:
    raw = vlm.extract_document_data(image_bytes, mime_type=mime_type)
    parsed, cleaned = parse_document_json(raw)
    return parsed, cleaned


def _manual_review_flags(parsed: Dict[str, Any]) -> Tuple[bool, List[str]]:
    reasons: List[str] = []
    threshold = _settings.review_confidence_threshold
    for entry in parsed.get("field_audit") or []:
        field = str(entry.get("field") or "")
        try:
            raw_score = entry.get("confidence_score")
            score = float(0.0 if raw_score is None else raw_score)
        except (TypeError, ValueError):
            score = 0.0
        if entry.get("is_critical") and score < threshold:
            reasons.append(f"{field} confidence {score:.2f} < {threshold:.2f}")
    return bool(reasons), reasons


def _coerce_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            text = value.strip()
            if not text:
                return None
            # If model accidentally outputs concatenated garbage, reject.
            if len(re.findall(r"\.", text)) > 1:
                return None
            return float(text)
    except (TypeError, ValueError):
        return None
    return None


def _sanitize_pii_regions(parsed: Dict[str, Any]) -> List[Dict[str, Any]]:
    sanitized: List[Dict[str, Any]] = []
    dropped = 0
    for region in parsed.get("pii_regions") or []:
        if not isinstance(region, dict):
            dropped += 1
            continue
        bbox_raw = region.get("bbox") or []
        if not isinstance(bbox_raw, list) or len(bbox_raw) != 4:
            dropped += 1
            continue
        bbox_vals: List[float] = []
        malformed = False
        for v in bbox_raw:
            fv = _coerce_float(v)
            if fv is None:
                malformed = True
                break
            bbox_vals.append(max(0.0, min(1.0, fv)))
        if malformed:
            dropped += 1
            continue

        c = _coerce_float(region.get("confidence_score"))
        sanitized.append(
            {
                "label": str(region.get("label") or "PII"),
                "value": region.get("value"),
                "bbox": bbox_vals,
                "confidence_score": 0.0 if c is None else max(0.0, min(1.0, c)),
            }
        )
    # #region agent log
    _dbg(
        "pre-fix",
        "H5",
        "main.py:_sanitize_pii_regions",
        "Sanitized pii regions",
        {
            "input_count": len(parsed.get("pii_regions") or []),
            "output_count": len(sanitized),
            "dropped_count": dropped,
        },
    )
    # #endregion
    return sanitized


def _to_document_data(parsed: Dict[str, Any], cleaned: str, image_bytes: bytes) -> DocumentData:
    requires_manual_review, review_reasons = _manual_review_flags(parsed)
    pii_regions = _sanitize_pii_regions(parsed)
    redacted_image_base64 = PrivacyService.redact_pii_regions(image_bytes, pii_regions)
    return DocumentData(
        document_type=str(parsed["document_type"]),
        name=parsed.get("name"),
        id_number=parsed.get("id_number"),
        date_of_birth=parsed.get("date_of_birth"),
        full_text=parsed.get("full_text"),
        summary=parsed.get("summary"),
        key_value_pairs=parsed.get("key_value_pairs") or [],
        entities=parsed.get("entities") or [],
        normalized_data=parsed.get("normalized_data") or {},
        field_audit=parsed.get("field_audit") or [],
        pii_regions=pii_regions,
        requires_manual_review=requires_manual_review,
        manual_review_reasons=review_reasons,
        redacted_image_base64=redacted_image_base64 or None,
        raw_json_string=cleaned,
    )


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/extract", response_model=DocumentData)
def extract(
    file: UploadFile = File(...),
    vlm: VLMService = Depends(get_vlm),
):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Upload an image file (e.g. image/jpeg, image/png).",
        )
    try:
        image_bytes = file.file.read()
    finally:
        file.file.close()

    if not image_bytes:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Empty file.")

    try:
        # #region agent log
        _dbg(
            "pre-fix",
            "H1",
            "main.py:extract:start",
            "Extract endpoint called",
            {"content_type": file.content_type or "none"},
        )
        # #endregion
        parsed, cleaned = _extract_or_error(vlm, image_bytes, file.content_type)
    except json.JSONDecodeError as e:
        body = ExtractErrorResponse(
            detail="Model output was not valid JSON.",
            raw_model_output=e.doc,
        ).model_dump()
        return JSONResponse(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, content=body)
    except Exception as e:
        logger.exception("Groq vision call failed")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Vision model request failed: {e!s}",
        ) from e

    if "document_type" not in parsed or parsed.get("document_type") in (None, ""):
        body = ExtractErrorResponse(
            detail='JSON must include non-empty "document_type".',
            raw_model_output=cleaned,
        ).model_dump()
        return JSONResponse(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, content=body)

    try:
        # #region agent log
        _dbg(
            "pre-fix",
            "H2",
            "main.py:extract:parsed",
            "Parsed model output before DocumentData",
            {
                "doc_type": str(parsed.get("document_type") or ""),
                "pii_regions_count": len(parsed.get("pii_regions") or []),
                "field_audit_count": len(parsed.get("field_audit") or []),
            },
        )
        # #endregion
        doc = _to_document_data(parsed, cleaned, image_bytes)
        # #region agent log
        _dbg(
            "pre-fix",
            "H3",
            "main.py:extract:success",
            "DocumentData created successfully",
            {
                "requires_manual_review": doc.requires_manual_review,
                "redacted_present": bool(doc.redacted_image_base64),
            },
        )
        # #endregion
        return doc
    except Exception as e:
        # #region agent log
        _dbg(
            "pre-fix",
            "H4",
            "main.py:extract:validation_error",
            "DocumentData validation failed",
            {"error": str(e)},
        )
        # #endregion
        body = ExtractErrorResponse(
            detail=f"Validation failed: {e!s}",
            raw_model_output=cleaned,
        ).model_dump()
        return JSONResponse(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, content=body)


@app.post("/verify-pair", response_model=PairVerificationResponse)
def verify_pair(
    file_a: UploadFile = File(...),
    file_b: UploadFile = File(...),
    vlm: VLMService = Depends(get_vlm),
):
    for f in (file_a, file_b):
        if not f.content_type or not f.content_type.startswith("image/"):
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Both uploads must be image files.")

    try:
        bytes_a = file_a.file.read()
        bytes_b = file_b.file.read()
    finally:
        file_a.file.close()
        file_b.file.close()

    if not bytes_a or not bytes_b:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Both files must be non-empty.")

    try:
        parsed_a, cleaned_a = _extract_or_error(vlm, bytes_a, file_a.content_type or "image/jpeg")
        parsed_b, cleaned_b = _extract_or_error(vlm, bytes_b, file_b.content_type or "image/jpeg")
    except json.JSONDecodeError as e:
        body = ExtractErrorResponse(
            detail="One of the model outputs was not valid JSON.",
            raw_model_output=e.doc,
        ).model_dump()
        return JSONResponse(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, content=body)
    except Exception as e:
        logger.exception("Pair verification extraction failed")
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=f"Verification extraction failed: {e!s}") from e

    doc_a = _to_document_data(parsed_a, cleaned_a, bytes_a)
    doc_b = _to_document_data(parsed_b, cleaned_b, bytes_b)

    report = [
        compare_field("name", doc_a.name, doc_b.name, threshold=0.86),
        compare_field("id_number", doc_a.id_number, doc_b.id_number, threshold=0.95),
        compare_field("date_of_birth", doc_a.date_of_birth, doc_b.date_of_birth, threshold=0.98),
        compare_field(
            "address",
            str(doc_a.normalized_data.get("address") or ""),
            str(doc_b.normalized_data.get("address") or ""),
            threshold=0.8,
        ),
    ]
    overall = sum(item.similarity_score for item in report) / len(report)
    discrepancies = [f.field for f in report if not f.is_match]
    return PairVerificationResponse(
        doc_a=doc_a,
        doc_b=doc_b,
        overall_match_score=round(overall, 4),
        is_match=(overall >= 0.88 and not discrepancies),
        match_report=report,
        discrepancies=discrepancies,
    )


_frontend_dist = Path(__file__).resolve().parent / "frontend" / "dist"
if _frontend_dist.is_dir():
    app.mount("/", StaticFiles(directory=str(_frontend_dist), html=True), name="frontend")
