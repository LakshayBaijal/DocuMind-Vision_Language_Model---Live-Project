# Benchmark folder

This folder contains the benchmark spec, a runnable scorer, and the expected label format.

## Step by step

1. Create this folder structure:

```text
Benchmark/
  data/
    images/
    labels/
```

2. Save each document image in `Benchmark/data/images/`.
3. Save one label JSON file per image in `Benchmark/data/labels/`.
4. Keep the image and label filenames aligned by stem:
   - `id_clean_01.jpg` -> `id_clean_01.json`
   - `passport_clean_01.png` -> `passport_clean_01.json`
5. Inspect what the script found:

```powershell
python Benchmark/run_benchmark.py --inspect
```

6. Run the benchmark against the deployed API:

```powershell
python Benchmark/run_benchmark.py --base-url https://documind-vision-language-model-live.onrender.com
```

7. Read the outputs:
   - `Benchmark/results/summary.json`
   - `Benchmark/results/summary.csv`
   - `Benchmark/results/predictions.jsonl`

## Label format

Use the structure in `Benchmark/label_template.json`.

## What the runner scores

- `document_type`
- `name`
- `id_number`
- `date_of_birth`
- `full_text` token F1
- `summary` factual checklist
- `key_value_pairs`
- `entities`
- `normalized_data`
- `requires_manual_review`
- `pii_regions`
- `redacted_image_base64`

## Notes

- The runner uses the same `/extract` API that the UI uses.
- If you want a browser-level UI check, compare the same file through the deployed site and the direct API output.
## No-label benchmark (fast proxy mode)

If labeling every image is expensive, you can benchmark without labels using quality proxy metrics.

1. Add images only to `Benchmark/data/images/`.
2. Inspect discovered images:

```powershell
python Benchmark/run_benchmark_unlabeled.py --inspect
```

3. Run unlabeled benchmark:

```powershell
python Benchmark/run_benchmark_unlabeled.py --base-url https://documind-vision-language-model-live.onrender.com
```

4. Read outputs:
   - `Benchmark/results/summary_unlabeled.json`
   - `Benchmark/results/summary_unlabeled.csv`
   - `Benchmark/results/predictions_unlabeled.jsonl`

Proxy metrics include schema validity, field presence, field-audit confidence, cross-field consistency, PII bbox validity, redaction presence, and a weighted `overall_quality_score` from `0` to `100`.

## Download documents automatically

If you have URLs for the documents, put one URL per line in `Benchmark/data/urls.txt` and run:

```powershell
python Benchmark/download_documents.py
```

You can also pass URLs directly:

```powershell
python Benchmark/download_documents.py https://example.com/doc1.jpg https://example.com/doc2.png
```

Downloaded files go into `Benchmark/data/images/`.
