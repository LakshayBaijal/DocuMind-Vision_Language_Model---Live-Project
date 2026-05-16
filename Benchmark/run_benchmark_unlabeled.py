from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
import sys
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import httpx

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models import DocumentData


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".gif"}
DEFAULT_BASE_URL = "https://documind-vision-language-model-live.onrender.com"
DEFAULT_DATA_DIR = Path(__file__).resolve().parent / "data"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "results"


@dataclass
class Fixture:
    stem: str
    image_path: Path


def normalize_text(value: Any) -> str:
    text = "" if value is None else str(value)
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"\s+", " ", text).strip().lower()
    return text


def normalize_identifier(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", normalize_text(value))


def discover_images(data_dir: Path) -> List[Fixture]:
    images_dir = data_dir / "images"
    search_root = images_dir if images_dir.is_dir() else data_dir
    image_paths = sorted(
        [path for path in search_root.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS]
    )
    return [Fixture(stem=path.stem, image_path=path) for path in image_paths]


def request_extract(client: httpx.Client, base_url: str, image_path: Path) -> Tuple[int, Dict[str, Any]]:
    mime = "image/jpeg"
    if image_path.suffix.lower() == ".png":
        mime = "image/png"
    elif image_path.suffix.lower() == ".webp":
        mime = "image/webp"
    elif image_path.suffix.lower() == ".gif":
        mime = "image/gif"

    with image_path.open("rb") as handle:
        files = {"file": (image_path.name, handle, mime)}
        response = client.post(f"{base_url.rstrip('/')}/extract", files=files)

    try:
        body = response.json()
    except Exception:
        body = {"detail": response.text}
    return response.status_code, body


def is_valid_bbox(bbox: Sequence[Any]) -> bool:
    if not isinstance(bbox, list) or len(bbox) != 4:
        return False
    try:
        x1, y1, x2, y2 = [float(v) for v in bbox]
    except (TypeError, ValueError):
        return False
    if x1 < 0 or y1 < 0 or x2 > 1 or y2 > 1:
        return False
    return x2 > x1 and y2 > y1


def safe_mean(values: List[float]) -> float:
    return statistics.fmean(values) if values else 0.0


def field_presence_score(result: DocumentData) -> float:
    checks = [
        bool(normalize_text(result.document_type)),
        bool(normalize_text(result.name)),
        bool(normalize_text(result.id_number)),
        bool(normalize_text(result.full_text)),
        bool(normalize_text(result.summary)),
    ]
    return sum(1 for item in checks if item) / len(checks)


def confidence_metrics(result: DocumentData, threshold: float) -> Dict[str, float]:
    if not result.field_audit:
        return {
            "mean_confidence": 0.0,
            "critical_low_conf_rate": 1.0,
            "audit_coverage": 0.0,
        }

    confidences = [max(0.0, min(1.0, float(row.confidence_score))) for row in result.field_audit]
    critical = [row for row in result.field_audit if bool(row.is_critical)]
    if critical:
        low_critical = sum(1 for row in critical if float(row.confidence_score) < threshold)
        critical_low_rate = low_critical / len(critical)
    else:
        critical_low_rate = 0.0

    return {
        "mean_confidence": safe_mean(confidences),
        "critical_low_conf_rate": critical_low_rate,
        "audit_coverage": 1.0,
    }


def consistency_score(result: DocumentData) -> Dict[str, float]:
    checks: List[float] = []

    normalized_name = normalize_text(result.normalized_data.get("name") if isinstance(result.normalized_data, dict) else None)
    top_name = normalize_text(result.name)
    if normalized_name and top_name:
        checks.append(1.0 if normalized_name == top_name else 0.0)

    normalized_id_candidates = []
    if isinstance(result.normalized_data, dict):
        for key in ("id_number", "student_id", "identifier", "id"):
            value = result.normalized_data.get(key)
            if value is not None:
                normalized_id_candidates.append(value)
    top_id = normalize_identifier(result.id_number)
    if top_id and normalized_id_candidates:
        id_match = any(normalize_identifier(candidate) == top_id for candidate in normalized_id_candidates)
        checks.append(1.0 if id_match else 0.0)

    kv_name = None
    for pair in result.key_value_pairs or []:
        key = normalize_text(pair.key)
        if key in {"name", "student name", "full name"}:
            kv_name = normalize_text(pair.value)
            break
    if top_name and kv_name:
        checks.append(1.0 if kv_name == top_name else 0.0)

    entity_name = None
    for entity in result.entities or []:
        if normalize_text(entity.label) in {"name", "person", "student name"}:
            entity_name = normalize_text(entity.value)
            break
    if top_name and entity_name:
        checks.append(1.0 if entity_name == top_name else 0.0)

    return {
        "consistency_score": safe_mean(checks) if checks else 0.0,
        "consistency_checks_count": float(len(checks)),
    }


def pii_metrics(result: DocumentData) -> Dict[str, float]:
    pii_regions = result.pii_regions or []
    if not pii_regions:
        return {
            "pii_bbox_valid_rate": 1.0,
            "redaction_when_pii": 1.0,
            "pii_count": 0.0,
        }

    valid_boxes = sum(1 for region in pii_regions if is_valid_bbox(region.bbox))
    bbox_valid_rate = valid_boxes / len(pii_regions)
    redaction_present = 1.0 if result.redacted_image_base64 else 0.0

    return {
        "pii_bbox_valid_rate": bbox_valid_rate,
        "redaction_when_pii": redaction_present,
        "pii_count": float(len(pii_regions)),
    }


def structure_metrics(result: DocumentData) -> Dict[str, float]:
    key_value_count = len(result.key_value_pairs or [])
    entities_count = len(result.entities or [])
    normalized_count = len(result.normalized_data or {}) if isinstance(result.normalized_data, dict) else 0
    non_empty_structures = sum(
        [
            1 if key_value_count > 0 else 0,
            1 if entities_count > 0 else 0,
            1 if normalized_count > 0 else 0,
        ]
    )
    return {
        "key_value_count": float(key_value_count),
        "entities_count": float(entities_count),
        "normalized_data_key_count": float(normalized_count),
        "structure_richness": non_empty_structures / 3.0,
    }


def compute_fixture_score(metrics: Dict[str, float]) -> float:
    parts = {
        "schema_valid": (metrics.get("schema_valid", 0.0), 0.25),
        "field_presence": (metrics.get("field_presence", 0.0), 0.20),
        "mean_confidence": (metrics.get("mean_confidence", 0.0), 0.20),
        "consistency_score": (metrics.get("consistency_score", 0.0), 0.20),
        "privacy_score": (
            (metrics.get("pii_bbox_valid_rate", 0.0) + metrics.get("redaction_when_pii", 0.0)) / 2.0,
            0.15,
        ),
    }
    score_0_1 = sum(value * weight for value, weight in parts.values())
    return score_0_1 * 100.0


def evaluate_fixture(client: httpx.Client, base_url: str, fixture: Fixture, confidence_threshold: float) -> Dict[str, Any]:
    status_code, body = request_extract(client, base_url, fixture.image_path)
    report: Dict[str, Any] = {
        "fixture": fixture.stem,
        "image": str(fixture.image_path),
        "status_code": status_code,
        "schema_valid": False,
        "errors": [],
        "metrics": {},
    }

    if status_code != 200:
        report["errors"].append(body.get("detail", f"HTTP {status_code}"))
        report["metrics"] = {
            "schema_valid": 0.0,
            "field_presence": 0.0,
            "mean_confidence": 0.0,
            "critical_low_conf_rate": 1.0,
            "audit_coverage": 0.0,
            "consistency_score": 0.0,
            "consistency_checks_count": 0.0,
            "pii_bbox_valid_rate": 0.0,
            "redaction_when_pii": 0.0,
            "pii_count": 0.0,
            "key_value_count": 0.0,
            "entities_count": 0.0,
            "normalized_data_key_count": 0.0,
            "structure_richness": 0.0,
        }
        report["overall_quality_score"] = compute_fixture_score(report["metrics"])
        return report

    try:
        parsed = DocumentData.model_validate(body)
        report["schema_valid"] = True
    except Exception as error:
        report["errors"].append(f"Schema validation failed: {error}")
        report["metrics"] = {
            "schema_valid": 0.0,
            "field_presence": 0.0,
            "mean_confidence": 0.0,
            "critical_low_conf_rate": 1.0,
            "audit_coverage": 0.0,
            "consistency_score": 0.0,
            "consistency_checks_count": 0.0,
            "pii_bbox_valid_rate": 0.0,
            "redaction_when_pii": 0.0,
            "pii_count": 0.0,
            "key_value_count": 0.0,
            "entities_count": 0.0,
            "normalized_data_key_count": 0.0,
            "structure_richness": 0.0,
        }
        report["overall_quality_score"] = compute_fixture_score(report["metrics"])
        return report

    metrics: Dict[str, float] = {}
    metrics["schema_valid"] = 1.0
    metrics["field_presence"] = field_presence_score(parsed)
    metrics.update(confidence_metrics(parsed, confidence_threshold))
    metrics.update(consistency_score(parsed))
    metrics.update(pii_metrics(parsed))
    metrics.update(structure_metrics(parsed))

    report["metrics"] = metrics
    report["overall_quality_score"] = compute_fixture_score(metrics)
    return report


def aggregate_reports(reports: List[Dict[str, Any]]) -> Dict[str, Any]:
    metric_names = sorted({name for report in reports for name in report.get("metrics", {}).keys()})
    aggregates: Dict[str, Dict[str, float]] = {}
    for metric_name in metric_names:
        values = [float(report["metrics"].get(metric_name, 0.0)) for report in reports]
        aggregates[metric_name] = {
            "mean": safe_mean(values),
            "min": min(values) if values else 0.0,
            "max": max(values) if values else 0.0,
        }

    quality_scores = [float(report.get("overall_quality_score", 0.0)) for report in reports]
    schema_valid_count = sum(1 for report in reports if report.get("schema_valid"))

    return {
        "total_fixtures": len(reports),
        "schema_valid_fixtures": schema_valid_count,
        "http_success_rate": safe_mean([1.0 if report.get("status_code") == 200 else 0.0 for report in reports]),
        "overall_quality_score_mean": safe_mean(quality_scores),
        "overall_quality_score_min": min(quality_scores) if quality_scores else 0.0,
        "overall_quality_score_max": max(quality_scores) if quality_scores else 0.0,
        "metric_aggregates": aggregates,
        "failed_fixtures": [report["fixture"] for report in reports if report.get("errors")],
        "fixtures": reports,
    }


def write_outputs(output_dir: Path, summary: Dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    (output_dir / "summary_unlabeled.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    with (output_dir / "summary_unlabeled.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["metric", "mean", "min", "max"],
        )
        writer.writeheader()
        for metric_name, aggregate in sorted(summary["metric_aggregates"].items()):
            writer.writerow(
                {
                    "metric": metric_name,
                    "mean": f"{aggregate['mean']:.4f}",
                    "min": f"{aggregate['min']:.4f}",
                    "max": f"{aggregate['max']:.4f}",
                }
            )

        writer.writerow({"metric": "overall_quality_score", "mean": f"{summary['overall_quality_score_mean']:.4f}", "min": f"{summary['overall_quality_score_min']:.4f}", "max": f"{summary['overall_quality_score_max']:.4f}"})
        writer.writerow({"metric": "http_success_rate", "mean": f"{summary['http_success_rate']:.4f}", "min": "", "max": ""})

    with (output_dir / "predictions_unlabeled.jsonl").open("w", encoding="utf-8") as handle:
        for fixture in summary["fixtures"]:
            handle.write(json.dumps(fixture, ensure_ascii=False) + "\n")


def print_inspection(fixtures: List[Fixture]) -> None:
    if not fixtures:
        print("No image fixtures found.")
        print("Place files in Benchmark/data/images/ and re-run with --inspect.")
        return

    print(f"Found {len(fixtures)} image fixture(s):")
    for fixture in fixtures:
        print(f"- {fixture.stem}: {fixture.image_path.name}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run DocuMind benchmark without labels (proxy quality metrics).")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help="Base URL of the API to benchmark")
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR), help="Benchmark data directory")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Directory for benchmark outputs")
    parser.add_argument("--inspect", action="store_true", help="List discovered images and exit")
    parser.add_argument("--timeout", type=float, default=120.0, help="HTTP timeout per request in seconds")
    parser.add_argument("--confidence-threshold", type=float, default=0.85, help="Threshold used for low-confidence critical fields")
    args = parser.parse_args()

    fixtures = discover_images(Path(args.data_dir))

    if args.inspect:
        print_inspection(fixtures)
        return 0

    if not fixtures:
        print("No fixtures found. Add files to Benchmark/data/images/ and run again.", file=sys.stderr)
        return 2

    with httpx.Client(timeout=args.timeout) as client:
        reports = [
            evaluate_fixture(client, args.base_url, fixture, args.confidence_threshold)
            for fixture in fixtures
        ]

    summary = aggregate_reports(reports)
    write_outputs(Path(args.output_dir), summary)

    print(f"Unlabeled benchmark written to: {Path(args.output_dir)}")
    print(f"Schema valid: {summary['schema_valid_fixtures']}/{summary['total_fixtures']}")
    print(f"Overall quality score (0-100): {summary['overall_quality_score_mean']:.2f}")
    for metric_name, aggregate in sorted(summary["metric_aggregates"].items()):
        print(f"{metric_name}: mean={aggregate['mean']:.4f} min={aggregate['min']:.4f} max={aggregate['max']:.4f}")
    if summary["failed_fixtures"]:
        print("Failed fixtures:", ", ".join(summary["failed_fixtures"]))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
