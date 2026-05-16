# DocuMind Vision Language Model (Visual Reader)

A full-stack AI-powered document extraction and verification system using FastAPI (Python) for the backend and React + Vite (TypeScript) for the frontend. It leverages LLMs (Groq, Llama) and vision models for extracting structured data from images of documents, with privacy redaction and pairwise verification.

---

## Features

- **Document Extraction:** Upload an image of a document and extract structured fields (name, ID, DOB, summary, etc.).
- **Privacy Redaction:** Detect and redact PII regions in images.
- **Pairwise Verification:** Compare two documents for identity verification.
- **Manual Review Flags:** Confidence-based review triggers.
- **Health Check Endpoint:** For deployment monitoring.
- **Frontend:** Modern React UI for uploads and results.
- **Backend:** FastAPI with modular services and CORS support.
- **Render Deployment:** Single service for both frontend and backend.

---

## Tech Stack & Tools

- **Backend:**  
  - Python 3.11+
  - FastAPI
  - Uvicorn
  - Pydantic & Pydantic Settings
  - Pillow (image processing)
  - OpenCV (opencv-python-headless)
  - Groq API (LLM & Vision)
  - Python-dotenv (env management)
  - httpx (HTTP client)
- **Frontend:**  
  - React 18
  - TypeScript
  - Vite
  - @vitejs/plugin-react
- **DevOps/Deployment:**  
  - Render.com (with `render.yaml`)
  - `runtime.txt` for Python version pinning
  - `.env` for secrets/config (not committed)
- **Other:**  
  - CORS middleware
  - Static file serving for frontend

---

## Project Structure

```
Visual Reader/
│
├── frontend/                # React + Vite frontend
│   ├── src/                 # React source code
│   ├── dist/                # Built static files (auto-generated)
│   ├── package.json         # Frontend dependencies
│   ├── vite.config.ts       # Vite config
│   └── ...                  # Other frontend files
│
├── main.py                  # FastAPI backend entry point
├── models.py                # Pydantic models
├── privacy_service.py       # PII redaction logic
├── verification_service.py  # Pairwise verification logic
├── vlm_service.py           # Vision/LLM service integration
├── config.py                # Settings loader
├── requirements.txt         # Python dependencies
├── render.yaml              # Render deployment config
├── runtime.txt              # Python version pinning
├── .env                     # Environment variables (not committed)
└── ...
```

---

## API Endpoints

- `GET /health`  
  Health check endpoint.

- `POST /extract`  
  Upload an image file. Returns extracted document data.

- `POST /verify-pair`  
  Upload two image files. Returns verification report.

---

## Environment Variables

Set these in your Render dashboard (do **not** commit `.env`):

```
GROQ_API_KEY=your_groq_api_key
LLM_PROVIDER=groq
LLM_MODEL=llama-3.1-8b-instant
GROQ_VISION_MODEL=meta-llama/llama-4-scout-17b-16e-instruct
LLM_MAX_TOKENS=2048
LLM_TEMPERATURE=0.2
GROQ_RESPONSE_JSON_MODE=true
REVIEW_CONFIDENCE_THRESHOLD=0.85
OCR_CONFIDENCE_THRESHOLD=0.25
MAX_KEYWORDS=30
YOLO_CONFIDENCE=0.35
```

---

## Local Development

**Backend:**
```sh
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt
uvicorn main:app --reload
```

**Frontend:**
```sh
cd frontend
npm install
npm run dev
```

---

## Deployment (Render.com)

1. Push your code to GitHub.
2. Add `runtime.txt` with `python-3.11.9` in the root.
3. Add your environment variables in the Render dashboard.
4. Use this `render.yaml`:
    ```yaml
    services:
      - type: web
        name: documind-vlm
        env: python
        buildCommand: |
          cd frontend && npm install && npm run build && cd ..
          pip install -r requirements.txt
        startCommand: uvicorn main:app --host 0.0.0.0 --port 10000
        plan: free
        envVars:
          - key: PORT
            value: 10000
    ```
5. Deploy! The backend will serve the built frontend at the root URL.

---

## Benchmark Results

A quality benchmark was run on **35 document images** (from Wikimedia Commons and PaddleOCR test datasets) to assess extraction performance. Results use **proxy metrics** (no manual labels required) to evaluate schema validity, field coverage, confidence levels, and privacy redaction.

### Overall Quality Summary

| Metric | Value |
|--------|-------|
| **Total Images Processed** | 35 |
| **Successfully Extracted** | 23 (65.7%) |
| **HTTP Success Rate** | 65.7% |
| **Mean Quality Score** | 41.18 / 100 |
| **Quality Score Range** | 0–84.27 |
| **Failed Extractions** | 12 (34.3%) |

### Quality Metrics Breakdown

Detailed aggregate statistics for 14 extraction quality indicators:

| Metric | Mean | Min | Max | Notes |
|--------|------|-----|-----|-------|
| **schema_valid** | 0.657 | 0.0 | 1.0 | % of valid DocumentData objects |
| **field_presence** | 0.451 | 0.0 | 1.0 | % of expected fields extracted |
| **mean_confidence** | 0.278 | 0.0 | 1.0 | Avg confidence across all fields |
| **critical_low_conf_rate** | 0.733 | 0.0 | 1.0 | % of fields with confidence < 0.5 |
| **audit_coverage** | 0.314 | 0.0 | 1.0 | % of fields with audit trails |
| **consistency_score** | 0.024 | 0.0 | 0.5 | Field-to-field consistency (0–1) |
| **entities_count** | 1.37 | 0.0 | 13.0 | Avg # of entities extracted |
| **key_value_count** | 2.63 | 0.0 | 13.0 | Avg # of key-value pairs |
| **normalized_data_key_count** | 5.57 | 0.0 | 15.0 | Avg # of normalized fields |
| **pii_bbox_valid_rate** | 0.636 | 0.0 | 1.0 | % of PII detections with valid bboxes |
| **redaction_when_pii** | 0.657 | 0.0 | 1.0 | % of documents with PII correctly redacted |
| **pii_count** | 0.314 | 0.0 | 4.0 | Avg PII regions per document |
| **structure_richness** | 0.562 | 0.0 | 1.0 | Normalized data field coverage |
| **consistency_checks_count** | 0.171 | 0.0 | 3.0 | Avg # of consistency validations |

### Failed Extractions

**12 documents failed** due to:
- **HTTP 500 errors** (validation failure): Malformed API responses or schema violations
- **Timeouts**: Large/multi-page documents exceeded Render's 60-second limit
- **Empty/Null fields**: VLM could not infer structured data from low-quality or non-document images

**Failed fixtures:**
- Foreign Observer identification badge in the 1989 Namibian election
- paddleocr_06_det_res_img_10_sast
- paddleocr_13_e2e_res_img_10_pgnet
- paddleocr_15_00111002
- paddleocr_24_french_0
- paddleocr_25_254
- paddleocr_26_img623
- paddleocr_27_img_12
- paddleocr_28_det_res_img623_ct
- paddleocr_29_en_3
- paddleocr_30_img_10_east_starnet
- Passport for a journey to France (1837)

### Top-Performing Extractions

**Highest quality score (84.27/100):** Francuska legitymacja pilota (Polish pilot license)
- Schema: ✅ Valid
- Fields: ✅ 100% coverage
- Confidence: ✅ 88% average
- PII Redaction: ✅ 3 regions detected & redacted
- Normalized Fields: 13 keys

**Other strong results (>70/100):**
- Mislatel CPCN: 76.0/100
- Luxembourg legitimation & proof of residency card: 52.0/100

### Interpretation

**Strengths:**
- ✅ **PII Detection & Redaction:** 65.7% of documents correctly identify and blur sensitive information
- ✅ **Schema Robustness:** 65.7% of responses pass Pydantic validation
- ✅ **Normalized Field Extraction:** 5.57 fields extracted on average (good for structured queries)

**Areas for Improvement:**
- ⚠️ **Field Confidence:** Mean confidence is only 27.8% (73.3% of fields have confidence < 0.5)
  - **Action:** Consider confidence-based filtering or multi-pass verification for production use
- ⚠️ **Completeness:** Only 45.1% of expected fields extracted on average
  - **Action:** Improve VLM prompt engineering to cover more field types
- ⚠️ **Consistency:** Low consistency between extracted and normalized fields
  - **Action:** Strengthen normalization heuristics in `vlm_service.py`

### How to Run Benchmarks

**Unlabeled Benchmark (Recommended):**
```powershell
cd Benchmark
python run_benchmark_unlabeled.py --base-url "https://documind-vision-language-model-live.onrender.com"
```
Results saved to `results/summary_unlabeled.json`, `summary_unlabeled.csv`, and `predictions_unlabeled.jsonl`.

**Labeled Benchmark (Requires Ground-Truth Labels):**
```powershell
cd Benchmark
python run_benchmark_unlabeled.py --base-url "https://documind-vision-language-model-live.onrender.com"
```
Requires JSON ground-truth files in `data/labels/` matching image names; computes F1/precision/recall per field.

**Download Test Documents:**
```powershell
cd Benchmark
python download_random_documents.py --count 30
```
Fetches 30 random document images from Wikimedia Commons and PaddleOCR.

---
