# DocuMind Extraction Benchmark

Scope: benchmark `POST /extract` only, through both paths:

- **API path:** direct HTTP upload to `/extract`
- **UI path:** browser upload through the deployed frontend, then compare the returned JSON with the API baseline for the same file

Latency is excluded.

## Benchmark corpus

Use one fixed labeled corpus of 12 image fixtures:

| Fixture | Type | Expected labels |
|---|---|---|
| `id_clean_01.jpg` | clean ID card | `document_type`, `name`, `id_number`, `date_of_birth` |
| `id_degraded_01.jpg` | skewed / compressed ID card | `document_type`, `name`, `id_number`, `date_of_birth` |
| `passport_clean_01.jpg` | clean passport page | `document_type`, `name`, `id_number`, `date_of_birth`, `full_text` |
| `certificate_clean_01.jpg` | clean certificate / marksheet | `document_type`, `name`, `summary`, `key_value_pairs`, `normalized_data` |
| `certificate_blur_01.jpg` | blurred certificate | `document_type`, `name`, `summary`, `key_value_pairs`, `normalized_data` |
| `form_handwritten_01.jpg` | handwritten form | `document_type`, `name`, `key_value_pairs`, `entities`, `full_text` |
| `invoice_dense_01.jpg` | dense invoice / bill | `document_type`, `key_value_pairs`, `entities`, `full_text`, `normalized_data` |
| `address_proof_01.jpg` | address proof / statement | `document_type`, `name`, `normalized_data.address`, `pii_regions` |
| `low_contrast_01.jpg` | low-contrast document | `document_type`, `full_text`, `summary` |
| `mixed_table_01.jpg` | mixed table / text layout | `document_type`, `key_value_pairs`, `entities`, `normalized_data` |
| `small_text_01.jpg` | small-text document | `document_type`, `full_text`, `entities` |
| `glare_rotated_01.jpg` | glare + rotation | `document_type`, `name`, `id_number`, `date_of_birth`, `pii_regions` |

## Normalization rules for scoring

- Compare strings case-insensitively after trimming whitespace.
- Normalize Unicode to NFKC before comparing.
- For `id_number`, remove spaces and punctuation before exact-match checks.
- For `date_of_birth`, compare ISO date if present; otherwise compare normalized date text.
- For `normalized_data`, score only keys listed in the fixture label sheet.
- For `full_text`, compare against the human transcript using token-level F1.
- For `pii_regions`, score a prediction as correct when IoU with a labeled region is at least `0.5`.

## Pass / fail thresholds

| Metric | Exact input set | Exact output to compare | Pass threshold |
|---|---|---|---|
| JSON/schema validity | all 12 fixtures | response parses to `DocumentData` shape | `12/12` pass |
| `document_type` accuracy | all 12 fixtures | returned `document_type` vs ground truth | `100%` overall |
| `name` exact-match accuracy | fixtures with labeled `name` (`id_clean_01`, `id_degraded_01`, `passport_clean_01`, `certificate_clean_01`, `certificate_blur_01`, `form_handwritten_01`, `invoice_dense_01`, `address_proof_01`, `glare_rotated_01`) | normalized `name` | `>= 90%` overall, `100%` on clean fixtures |
| `id_number` exact-match accuracy | fixtures with labeled identifier (`id_clean_01`, `id_degraded_01`, `passport_clean_01`, `glare_rotated_01`) | normalized `id_number` | `>= 95%` overall, `100%` on clean fixtures |
| `date_of_birth` exact-match accuracy | fixtures with labeled DOB (`id_clean_01`, `id_degraded_01`, `passport_clean_01`, `glare_rotated_01`) | normalized `date_of_birth` | `>= 95%` overall, `100%` on clean fixtures |
| `full_text` fidelity | `passport_clean_01`, `form_handwritten_01`, `invoice_dense_01`, `low_contrast_01`, `mixed_table_01`, `small_text_01`, `glare_rotated_01`, `certificate_clean_01` | token-level F1 against transcript | `>= 0.90` on clean fixtures, `>= 0.75` on degraded fixtures |
| `summary` factuality | fixtures with ground-truth summary checklist (`certificate_clean_01`, `certificate_blur_01`, `invoice_dense_01`, `mixed_table_01`) | checklist of required facts, no hallucinations | all required facts present; zero unsupported facts |
| `key_value_pairs` extraction | `certificate_clean_01`, `certificate_blur_01`, `form_handwritten_01`, `invoice_dense_01`, `mixed_table_01`, `address_proof_01` | micro precision / recall / F1 on labeled pairs | `F1 >= 0.85` overall, `>= 0.75` on hard fixtures |
| `entities` extraction | all fixtures with labeled entities | micro precision / recall / F1 | `F1 >= 0.80` overall |
| `normalized_data` completeness | fixtures with normalized labels | required labeled keys present and correct | `>= 90%` required-key coverage, `>= 85%` exact value match |
| Manual-review flag quality | fixtures with critical-field labels | `requires_manual_review` vs ground truth | precision `>= 0.90`, recall `>= 0.90` |
| PII region detection | `address_proof_01`, `glare_rotated_01`, `id_clean_01`, `id_degraded_01`, `passport_clean_01` | predicted `pii_regions` vs labeled regions | region recall `>= 0.85`, precision `>= 0.75` |
| Redaction correctness | same fixtures as PII region detection | `redacted_image_base64` visually hides labeled PII | `100%` of labeled PII regions redacted |
| UI/API parity | all 12 fixtures through browser upload | browser-returned JSON vs direct API JSON baseline | `100%` field parity for all compared fields |

## Overall release gate

The benchmark passes only if:

1. JSON/schema validity is `12/12`.
2. Every row in the table meets its threshold.
3. UI/API parity is `100%` on all fixtures.

## Notes

- This benchmark intentionally ignores latency and throughput.
- If you want a single score, use a weighted blend after the pass/fail gate is satisfied.
- The frontend uses the same `/extract` endpoint, so the UI benchmark should be treated as a transport/parity check over the same extraction model output.

## How to calculate it

Use the scripts in this folder:

1. Put your test images in `Benchmark/data/images/`.
2. Put one JSON label file per image in `Benchmark/data/labels/` using the same file stem.
3. Run `python Benchmark/run_benchmark.py --inspect` to find which benchmark files are present.
4. Run `python Benchmark/run_benchmark.py --base-url https://documind-vision-language-model-live.onrender.com` to score the fixtures.
5. Read `Benchmark/results/summary.json` and `Benchmark/results/summary.csv` for the metrics.

The script scores extraction quality only. It does not measure latency.