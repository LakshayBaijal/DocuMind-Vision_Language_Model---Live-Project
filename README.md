# DocuMind Vision Language Model (Visual Reader)

A full-stack AI-powered document extraction and verification system using FastAPI (Python) for the backend and React + Vite (TypeScript) for the frontend. It leverages LLMs (Groq, Llama) and vision models for extracting structured data from images of documents, with privacy redaction and pairwise verification.

---


## Live Project Link 
```br
https://documind-vision-language-model-live.onrender.com/
```

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

