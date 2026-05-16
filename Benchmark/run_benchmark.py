from __future__ import annotations

import argparse
import base64
import csv
import io
import json
import math
import re
import sys
import unicodedata
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import httpx
from PIL import Image

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
    label_path: Optional[Path]
    labels: Dict[str, Any]


def normalize_text(value: Any) -> str:
    text = "" if value is None else str(value)
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"\s+", " ", text).strip().lower()
    return text


def normalize_identifier(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", normalize_text(value))


def tokenize(text: Any) -> List[str]:
    return re.findall(r"\w+", normalize_text(text), flags=re.UNICODE)


def f1_from_counts(correct: int, predicted: int, expected: int) -> Dict[str, float]:
    precision = correct / predicted if predicted else 0.0
    recall = correct / expected if expected else 0.0
    if precision + recall:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def flatten_dict(value: Any, prefix: str = "") -> Dict[str, Any]:
    flat: Dict[str, Any] = {}
    if isinstance(value, dict):
        for key, item in value.items():
            next_prefix = f"{prefix}.{key}" if prefix else str(key)
            flat.update(flatten_dict(item, next_prefix))
    elif isinstance(value, list):
        flat[prefix] = value
    else:
        flat[prefix] = value
    return flat


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def item_value(item: Any, key: str) -> Any:
    if isinstance(item, dict):
        return item.get(key)
    return getattr(item, key, None)


def discover_fixtures(data_dir: Path) -> List[Fixture]:
    images_dir = data_dir / "images"
    labels_dir = data_dir / "labels"

    search_root = images_dir if images_dir.is_dir() else data_dir
    image_paths = sorted(
        [p for p in search_root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS]
    )

    fixtures: List[Fixture] = []
    for image_path in image_paths:
        stem = image_path.stem
        label_path = labels_dir / f"{stem}.json"
        labels: Dict[str, Any] = {}
        if label_path.is_file():
            try:
                labels = load_json(label_path)
            except json.JSONDecodeError:
                labels = {"_label_error": f"Invalid JSON in {label_path.name}"}
        fixtures.append(Fixture(stem=stem, image_path=image_path, label_path=label_path if label_path.is_file() else None, labels=labels))
    return fixtures


def load_image(path: Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def crop_bbox(image: Image.Image, bbox: Sequence[Any]) -> Optional[Image.Image]:
    if len(bbox) != 4:
        return None
    try:
        x1, y1, x2, y2 = [float(v) for v in bbox]
    except (TypeError, ValueError):
        return None
    width, height = image.size
    left = max(0, min(width - 1, int(x1 * width)))
    top = max(0, min(height - 1, int(y1 * height)))
    right = max(left + 1, min(width, int(x2 * width)))
    bottom = max(top + 1, min(height, int(y2 * height)))
    if right <= left or bottom <= top:
        return None
    return image.crop((left, top, right, bottom))


def compute_iou(a: Sequence[Any], b: Sequence[Any]) -> float:
    if len(a) != 4 or len(b) != 4:
        return 0.0
    try:
        ax1, ay1, ax2, ay2 = [float(v) for v in a]
        bx1, by1, bx2, by2 = [float(v) for v in b]
    except (TypeError, ValueError):
        return 0.0

    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter_w = max(0.0, ix2 - ix1)
    inter_h = max(0.0, iy2 - iy1)
    inter = inter_w * inter_h
    if inter == 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    denom = area_a + area_b - inter
    return inter / denom if denom else 0.0


def match_multiset(expected: Sequence[Tuple[str, str]], predicted: Sequence[Tuple[str, str]]) -> Tuple[int, int, int]:
    expected_counter = Counter(expected)
    predicted_counter = Counter(predicted)
    correct = sum((expected_counter & predicted_counter).values())
    return correct, sum(predicted_counter.values()), sum(expected_counter.values())


def compare_fields(result: DocumentData, labels: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    metrics: Dict[str, Dict[str, Any]] = {}
    fields = labels.get("fields") or {}

    for field_name in ("document_type", "name", "id_number", "date_of_birth"):
        if field_name not in labels and field_name not in fields:
            continue
        expected = labels.get(field_name, fields.get(field_name))
        predicted = getattr(result, field_name)
        if field_name == "id_number":
            score = normalize_identifier(expected) == normalize_identifier(predicted)
        else:
            score = normalize_text(expected) == normalize_text(predicted)
        metrics[field_name] = {
            "correct": 1 if score else 0,
            "predicted": 1,
            "expected": 1,
            "pass": bool(score),
            "expected_value": expected,
            "predicted_value": predicted,
        }
    return metrics


def compare_full_text(result: DocumentData, labels: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    expected = labels.get("full_text") or labels.get("transcript")
    if expected is None:
        return None
    expected_tokens = tokenize(expected)
    predicted_tokens = tokenize(result.full_text)
    expected_counter = Counter(expected_tokens)
    predicted_counter = Counter(predicted_tokens)
    correct = sum((expected_counter & predicted_counter).values())
    stats = f1_from_counts(correct, sum(predicted_counter.values()), sum(expected_counter.values()))
    return {
        "correct": correct,
        "predicted": sum(predicted_counter.values()),
        "expected": sum(expected_counter.values()),
        **stats,
    }


def compare_summary(result: DocumentData, labels: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    facts = labels.get("summary_facts")
    if not facts:
        return None
    summary_text = normalize_text(result.summary)
    matches = sum(1 for fact in facts if normalize_text(fact) in summary_text)
    return {
        "correct": matches,
        "predicted": len(facts),
        "expected": len(facts),
        **f1_from_counts(matches, len(facts), len(facts)),
        "missing_facts": [fact for fact in facts if normalize_text(fact) not in summary_text],
    }


def compare_kv_or_entities(result_items: Sequence[Dict[str, Any]], expected_items: Sequence[Dict[str, Any]], key_a: str, key_b: str) -> Optional[Dict[str, Any]]:
    if not expected_items:
        return None
    predicted = [(normalize_text(item_value(item, key_a)), normalize_text(item_value(item, key_b))) for item in result_items or []]
    expected = [(normalize_text(item_value(item, key_a)), normalize_text(item_value(item, key_b))) for item in expected_items or []]
    correct, predicted_count, expected_count = match_multiset(expected, predicted)
    stats = f1_from_counts(correct, predicted_count, expected_count)
    return {"correct": correct, "predicted": predicted_count, "expected": expected_count, **stats}


def compare_normalized_data(result: DocumentData, labels: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    expected = labels.get("normalized_data")
    if not expected:
        return None
    expected_flat = flatten_dict(expected)
    predicted_flat = flatten_dict(result.normalized_data or {})
    correct = 0
    exact_matches = []
    missing = []
    for path, expected_value in expected_flat.items():
        predicted_value = predicted_flat.get(path)
        if isinstance(expected_value, (dict, list)):
            continue
        if normalize_text(predicted_value) == normalize_text(expected_value):
            correct += 1
            exact_matches.append(path)
        else:
            missing.append(path)
    expected_count = len([v for v in expected_flat.values() if not isinstance(v, (dict, list))])
    return {
        "correct": correct,
        "predicted": expected_count,
        "expected": expected_count,
        **f1_from_counts(correct, expected_count, expected_count),
        "matched_paths": exact_matches,
        "missing_paths": missing,
        "coverage": (correct / expected_count) if expected_count else 0.0,
    }


def compare_manual_review(result: DocumentData, labels: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if "manual_review" not in labels:
        return None
    expected = bool(labels.get("manual_review"))
    predicted = bool(result.requires_manual_review)
    correct = 1 if expected == predicted else 0
    return {
        "correct": correct,
        "predicted": 1,
        "expected": 1,
        "pass": bool(correct),
        "expected_value": expected,
        "predicted_value": predicted,
    }


def compare_pii_regions(result: DocumentData, labels: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    expected = labels.get("pii_regions") or []
    if not expected:
        return None
    predicted = list(result.pii_regions or [])
    used_predicted: set[int] = set()
    matches = 0
    match_details: List[Dict[str, Any]] = []
    for exp in expected:
        exp_bbox = exp.get("bbox") or []
        best_iou = 0.0
        best_idx = None
        for idx, pred in enumerate(predicted):
            if idx in used_predicted:
                continue
            iou = compute_iou(exp_bbox, pred.bbox or [])
            if iou > best_iou:
                best_iou = iou
                best_idx = idx
        if best_idx is not None and best_iou >= 0.5:
            used_predicted.add(best_idx)
            matches += 1
            match_details.append({"expected": exp_bbox, "predicted": predicted[best_idx].bbox, "iou": best_iou})
    return {
        "correct": matches,
        "predicted": len(predicted),
        "expected": len(expected),
        **f1_from_counts(matches, len(predicted), len(expected)),
        "matches": match_details,
    }


def compare_redaction(result: DocumentData, labels: Dict[str, Any], original_image: Path) -> Optional[Dict[str, Any]]:
    expected = labels.get("pii_regions") or []
    if not expected or not result.redacted_image_base64:
        return None
    try:
        original = load_image(original_image)
        redacted_bytes = base64.b64decode(result.redacted_image_base64)
        redacted = Image.open(io.BytesIO(redacted_bytes)).convert("RGB")
    except Exception:
        return None

    detected = 0
    details = []
    for region in expected:
        bbox = region.get("bbox") or []
        orig_crop = crop_bbox(original, bbox)
        red_crop = crop_bbox(redacted, bbox)
        if orig_crop is None or red_crop is None:
            continue
        orig_pixels = list(orig_crop.getdata())
        red_pixels = list(red_crop.getdata())
        if not orig_pixels or len(orig_pixels) != len(red_pixels):
            continue
        diffs = []
        for a, b in zip(orig_pixels, red_pixels):
            diffs.extend([abs(a[0] - b[0]), abs(a[1] - b[1]), abs(a[2] - b[2])])
        mean_diff = sum(diffs) / len(diffs)
        orig_std = _rgb_std(orig_pixels)
        red_std = _rgb_std(red_pixels)
        redacted_ok = mean_diff >= 2.0 or red_std <= orig_std * 0.9
        detected += 1 if redacted_ok else 0
        details.append({"bbox": bbox, "mean_diff": mean_diff, "orig_std": orig_std, "red_std": red_std, "redacted": redacted_ok})
    return {
        "correct": detected,
        "predicted": len(expected),
        "expected": len(expected),
        **f1_from_counts(detected, len(expected), len(expected)),
        "details": details,
    }


def _rgb_std(pixels: Sequence[Tuple[int, int, int]]) -> float:
    if not pixels:
        return 0.0
    channels = list(zip(*pixels))
    variances = []
    for channel in channels:
        mean = sum(channel) / len(channel)
        variance = sum((value - mean) ** 2 for value in channel) / len(channel)
        variances.append(variance)
    return math.sqrt(sum(variances) / len(variances))


def request_extract(client: httpx.Client, base_url: str, image_path: Path) -> Dict[str, Any]:
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

    body: Dict[str, Any]
    try:
        body = response.json()
    except Exception:
        body = {"detail": response.text}

    return {"status_code": response.status_code, "body": body}


def evaluate_fixture(client: httpx.Client, base_url: str, fixture: Fixture) -> Dict[str, Any]:
    result = request_extract(client, base_url, fixture.image_path)
    body = result["body"]
    fixture_report: Dict[str, Any] = {
        "fixture": fixture.stem,
        "image": str(fixture.image_path),
        "label": str(fixture.label_path) if fixture.label_path else None,
        "status_code": result["status_code"],
        "schema_valid": False,
        "metrics": {},
        "errors": [],
    }

    if result["status_code"] != 200:
        fixture_report["errors"].append(body.get("detail", f"HTTP {result['status_code']}"))
        return fixture_report

    try:
        parsed = DocumentData.model_validate(body)
        fixture_report["schema_valid"] = True
    except Exception as exc:
        fixture_report["errors"].append(f"Schema validation failed: {exc}")
        return fixture_report

    labels = fixture.labels
    fixture_report["metrics"].update(compare_fields(parsed, labels))

    full_text = compare_full_text(parsed, labels)
    if full_text is not None:
        fixture_report["metrics"]["full_text"] = full_text

    summary = compare_summary(parsed, labels)
    if summary is not None:
        fixture_report["metrics"]["summary"] = summary

    kv = compare_kv_or_entities(parsed.key_value_pairs, labels.get("key_value_pairs") or [], "key", "value")
    if kv is not None:
        fixture_report["metrics"]["key_value_pairs"] = kv

    entities = compare_kv_or_entities(parsed.entities, labels.get("entities") or [], "label", "value")
    if entities is not None:
        fixture_report["metrics"]["entities"] = entities

    normalized = compare_normalized_data(parsed, labels)
    if normalized is not None:
        fixture_report["metrics"]["normalized_data"] = normalized

    review = compare_manual_review(parsed, labels)
    if review is not None:
        fixture_report["metrics"]["manual_review"] = review

    pii = compare_pii_regions(parsed, labels)
    if pii is not None:
        fixture_report["metrics"]["pii_regions"] = pii

    redaction = compare_redaction(parsed, labels, fixture.image_path)
    if redaction is not None:
        fixture_report["metrics"]["redaction"] = redaction

    return fixture_report


def aggregate_metric(metric_reports: List[Dict[str, Any]]) -> Dict[str, Any]:
    correct = sum(int(item.get("correct", 0)) for item in metric_reports)
    predicted = sum(int(item.get("predicted", 0)) for item in metric_reports)
    expected = sum(int(item.get("expected", 0)) for item in metric_reports)
    stats = f1_from_counts(correct, predicted, expected)
    return {"correct": correct, "predicted": predicted, "expected": expected, **stats}


def summarize(reports: List[Dict[str, Any]], fixtures: List[Fixture]) -> Dict[str, Any]:
    by_metric: Dict[str, List[Dict[str, Any]]] = {}
    schema_valid = sum(1 for report in reports if report["schema_valid"])
    failures = [report for report in reports if report["errors"]]

    for report in reports:
        for metric_name, metric_report in report["metrics"].items():
            by_metric.setdefault(metric_name, []).append(metric_report)

    summary = {
        "total_fixtures": len(fixtures),
        "schema_valid_fixtures": schema_valid,
        "failed_fixtures": [item["fixture"] for item in failures],
        "metrics": {name: aggregate_metric(items) for name, items in by_metric.items()},
        "fixtures": reports,
    }
    return summary


def write_outputs(output_dir: Path, summary: Dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    csv_rows = []
    for metric_name, metric in sorted(summary["metrics"].items()):
        csv_rows.append(
            {
                "metric": metric_name,
                "correct": metric.get("correct", 0),
                "predicted": metric.get("predicted", 0),
                "expected": metric.get("expected", 0),
                "precision": f"{metric.get('precision', 0.0):.4f}",
                "recall": f"{metric.get('recall', 0.0):.4f}",
                "f1": f"{metric.get('f1', 0.0):.4f}",
            }
        )

    with (output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["metric", "correct", "predicted", "expected", "precision", "recall", "f1"])
        writer.writeheader()
        writer.writerows(csv_rows)

    with (output_dir / "predictions.jsonl").open("w", encoding="utf-8") as handle:
        for fixture in summary["fixtures"]:
            handle.write(json.dumps(fixture, ensure_ascii=False) + "\n")


def print_inspection(fixtures: List[Fixture]) -> None:
    if not fixtures:
        print("No image fixtures found.")
        print("Place files in Benchmark/data/images/ and matching labels in Benchmark/data/labels/.")
        return

    print(f"Found {len(fixtures)} image fixture(s):")
    for fixture in fixtures:
        label_status = "label ok" if fixture.label_path else "missing label"
        print(f"- {fixture.stem}: {fixture.image_path.name} ({label_status})")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the DocuMind extraction benchmark.")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help="Base URL of the API to benchmark")
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR), help="Benchmark data directory")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Directory for benchmark outputs")
    parser.add_argument("--inspect", action="store_true", help="List discovered fixtures and exit")
    parser.add_argument("--timeout", type=float, default=120.0, help="HTTP timeout per request in seconds")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    fixtures = discover_fixtures(data_dir)

    if args.inspect:
        print_inspection(fixtures)
        return 0

    if not fixtures:
        print("No fixtures found. Use --inspect after placing files in Benchmark/data/images/.", file=sys.stderr)
        return 2

    with httpx.Client(timeout=args.timeout) as client:
        reports = [evaluate_fixture(client, args.base_url, fixture) for fixture in fixtures]

    summary = summarize(reports, fixtures)
    write_outputs(output_dir, summary)

    print(f"Benchmarks written to: {output_dir}")
    print(f"Schema valid: {summary['schema_valid_fixtures']}/{summary['total_fixtures']}")
    for metric_name, metric in sorted(summary["metrics"].items()):
        print(f"{metric_name}: F1={metric['f1']:.4f} P={metric['precision']:.4f} R={metric['recall']:.4f}")
    if summary["failed_fixtures"]:
        print("Failed fixtures:", ", ".join(summary["failed_fixtures"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())