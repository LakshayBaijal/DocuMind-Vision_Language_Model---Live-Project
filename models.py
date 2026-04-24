from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field


class ExtractedPair(BaseModel):
    key: str
    value: str


class ExtractedEntity(BaseModel):
    label: str
    value: str


class FieldAudit(BaseModel):
    field: str
    value: Optional[str] = None
    confidence_score: float = Field(ge=0.0, le=1.0)
    reasoning: str
    is_critical: bool = False


class PIIRegion(BaseModel):
    label: str
    value: Optional[str] = None
    bbox: List[float] = Field(
        default_factory=list,
        description="Normalized [x1,y1,x2,y2] in 0..1 coordinates",
    )
    confidence_score: float = Field(default=0.0, ge=0.0, le=1.0)


class DocumentData(BaseModel):
    """Rich structured document payload parsed from the VLM JSON response."""

    model_config = ConfigDict(extra="ignore")

    document_type: str = Field(..., description="Detected document category")
    name: Optional[str] = None
    id_number: Optional[str] = None
    date_of_birth: Optional[str] = None
    full_text: Optional[str] = Field(
        default=None,
        description="Best effort transcription of all visible text in reading order",
    )
    summary: Optional[str] = Field(
        default=None,
        description="Short human-readable interpretation of the document contents",
    )
    key_value_pairs: List[ExtractedPair] = Field(default_factory=list)
    entities: List[ExtractedEntity] = Field(default_factory=list)
    normalized_data: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional normalized structured details inferred by the LLM",
    )
    field_audit: List[FieldAudit] = Field(default_factory=list)
    pii_regions: List[PIIRegion] = Field(default_factory=list)
    requires_manual_review: bool = False
    manual_review_reasons: List[str] = Field(default_factory=list)
    redacted_image_base64: Optional[str] = None
    raw_json_string: str = Field(..., description="Exact model output before parsing")


class ExtractErrorResponse(BaseModel):
    detail: str
    raw_model_output: Optional[str] = None


class VerificationField(BaseModel):
    field: str
    value_a: Optional[str] = None
    value_b: Optional[str] = None
    similarity_score: float = Field(ge=0.0, le=1.0)
    is_match: bool


class PairVerificationResponse(BaseModel):
    doc_a: DocumentData
    doc_b: DocumentData
    overall_match_score: float = Field(ge=0.0, le=1.0)
    is_match: bool
    match_report: List[VerificationField]
    discrepancies: List[str] = Field(default_factory=list)
