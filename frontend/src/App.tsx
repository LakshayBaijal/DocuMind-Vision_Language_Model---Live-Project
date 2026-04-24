import { useCallback, useEffect, useRef, useState } from "react";

type DocumentResult = {
  document_type: string;
  name: string | null;
  id_number: string | null;
  date_of_birth: string | null;
  full_text: string | null;
  summary: string | null;
  key_value_pairs: Array<{ key: string; value: string }>;
  entities: Array<{ label: string; value: string }>;
  normalized_data: Record<string, unknown>;
  field_audit: Array<{
    field: string;
    value: string | null;
    confidence_score: number;
    reasoning: string;
    is_critical: boolean;
  }>;
  requires_manual_review: boolean;
  manual_review_reasons: string[];
  redacted_image_base64: string | null;
  raw_json_string: string;
};

type ErrorBody = {
  detail: string;
  raw_model_output?: string | null;
};

const API = "";
type BaseFieldKey = "document_type" | "name" | "id_number" | "date_of_birth";
type SensitiveFieldKey = "id_number" | "date_of_birth";

function agentDebugLog(runId: string, hypothesisId: string, location: string, message: string, data: Record<string, unknown>) {
  fetch("http://127.0.0.1:7356/ingest/2d13224f-3c29-48ad-af7b-cd13615831fc", {
    method: "POST",
    headers: { "Content-Type": "application/json", "X-Debug-Session-Id": "a9b039" },
    body: JSON.stringify({
      sessionId: "a9b039",
      runId,
      hypothesisId,
      location,
      message,
      data,
      timestamp: Date.now(),
    }),
  }).catch(() => {});
}

function formatLabel(key: string): string {
  return key.replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase());
}

const BASE_FIELDS: BaseFieldKey[] = ["document_type", "name", "id_number", "date_of_birth"];
const SENSITIVE_FIELDS: SensitiveFieldKey[] = ["id_number", "date_of_birth"];

function maskValue(value: string | null | undefined): string {
  if (!value) return "—";
  const visible = 2;
  const clean = String(value);
  if (clean.length <= visible * 2) return "*".repeat(clean.length);
  return `${clean.slice(0, visible)}${"*".repeat(clean.length - visible * 2)}${clean.slice(clean.length - visible)}`;
}

export function App() {
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<DocumentResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [rawOpen, setRawOpen] = useState(false);
  const [hideSensitive, setHideSensitive] = useState(true);
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    if (!file) {
      setPreview(null);
      return;
    }
    const url = URL.createObjectURL(file);
    setPreview(url);
    return () => URL.revokeObjectURL(url);
  }, [file]);

  const onFile = useCallback((f: File | null) => {
    setError(null);
    setResult(null);
    setRawOpen(false);
    setHideSensitive(true);
    if (!f || !f.type.startsWith("image/")) {
      setFile(null);
      if (f) setError("Please choose an image file (JPEG, PNG, or WebP).");
      return;
    }
    setFile(f);
  }, []);

  const extract = useCallback(async () => {
    if (!file) return;
    setLoading(true);
    setError(null);
    setResult(null);
    // #region agent log
    agentDebugLog("pre-fix", "H1", "App.tsx:extract:start", "Extract started", {
      hasFile: Boolean(file),
      fileType: file.type || "unknown",
      hideSensitive,
    });
    // #endregion
    try {
      const fd = new FormData();
      fd.append("file", file, file.name);
      // #region agent log
      agentDebugLog("pre-fix", "H2", "App.tsx:extract:beforeFetch", "Calling /extract API", {
        fileNameLength: file.name.length,
      });
      // #endregion
      const res = await fetch(`${API}/extract`, { method: "POST", body: fd });
      const data = await res.json().catch(() => ({}));
      // #region agent log
      agentDebugLog("pre-fix", "H3", "App.tsx:extract:afterFetch", "Received /extract response", {
        status: res.status,
        ok: res.ok,
        hasDetail: Boolean((data as ErrorBody).detail),
      });
      // #endregion
      if (!res.ok) {
        const err = data as ErrorBody;
        const msg = err.detail || res.statusText || "Request failed";
        const extra = err.raw_model_output
          ? `\n\nModel output:\n${err.raw_model_output}`
          : "";
        setError(msg + extra);
        return;
      }
      setResult(data as DocumentResult);
    } catch (e) {
      // #region agent log
      agentDebugLog("pre-fix", "H4", "App.tsx:extract:catch", "Extract failed in frontend", {
        errorType: e instanceof Error ? e.name : typeof e,
      });
      // #endregion
      setError(e instanceof Error ? e.message : "Network error");
    } finally {
      setLoading(false);
    }
  }, [file, hideSensitive]);

  const onDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      const f = e.dataTransfer.files[0];
      onFile(f ?? null);
    },
    [onFile]
  );

  return (
    <div className="page">
      <header className="header">
        <div className="brand">
          <span className="logo" aria-hidden />
          <div>
            <h1>DocuMind VLM</h1>
            <p className="tagline">Focuses on the end goal: turning raw pixels into a structured schema.</p>
          </div>
        </div>
        <p className="subtag">
          Upload any document image and extract comprehensive machine-readable data in one shot.
        </p>
      </header>

      <main className="main">
        <section
          className={`dropzone ${file ? "has-file" : ""}`}
          onDragOver={(e) => e.preventDefault()}
          onDrop={onDrop}
          onClick={() => inputRef.current?.click()}
          role="button"
          tabIndex={0}
          onKeyDown={(e) => {
            if (e.key === "Enter" || e.key === " ") {
              e.preventDefault();
              inputRef.current?.click();
            }
          }}
        >
          <input
            ref={inputRef}
            type="file"
            accept="image/jpeg,image/png,image/webp,image/gif"
            className="hidden-input"
            onChange={(e) => onFile(e.target.files?.[0] ?? null)}
          />
          {!file ? (
            <div className="dropzone-inner">
              <p className="drop-title">Drop a document image here</p>
              <p className="drop-hint">or click to browse - IDs, certificates, forms, and reports</p>
            </div>
          ) : (
            <div className="preview-wrap">
              {preview && (
                <img
                  src={preview}
                  alt="Selected document"
                  className={`preview-img ${hideSensitive ? "preview-sensitive" : ""}`}
                />
              )}
              <p className="file-name">{file.name}</p>
            </div>
          )}
        </section>

        <div className="actions">
          <div className="pre-extract-privacy">
            <label className="sensitive-toggle">
              <input
                type="checkbox"
                checked={hideSensitive}
                onChange={(e) => setHideSensitive(e.target.checked)}
                disabled={!file || loading}
              />
              <span>Hide sensitive data before extraction</span>
            </label>
            <span className="privacy-note">Preview and displayed fields are masked before extraction starts.</span>
            <span className="privacy-note">Extraction still uses the original image for best accuracy.</span>
          </div>
          <button type="button" className="btn secondary" onClick={() => onFile(null)} disabled={!file || loading}>
            Clear
          </button>
          <button type="button" className="btn primary" onClick={extract} disabled={!file || loading}>
            {loading ? "Extracting..." : "Extract into Structured Schema"}
          </button>
        </div>

        {error && (
          <div className="panel error-panel" role="alert">
            <h2>Something went wrong</h2>
            <pre className="error-text">{error}</pre>
          </div>
        )}

        {result && (
          <section className="panel results">
            <h2>Structured Schema Output</h2>
            <p className="results-note">
              Groq vision + LLM normalization converted this document into detailed, structured JSON.
            </p>
            <dl className="field-grid">
              {BASE_FIELDS.map((key) => {
                const value = result[key];
                const displayValue =
                  hideSensitive && SENSITIVE_FIELDS.includes(key as SensitiveFieldKey)
                    ? maskValue(value)
                    : value ?? "—";
                return (
                  <div key={key} className="field-row">
                    <dt>{formatLabel(String(key))}</dt>
                    <dd>{displayValue}</dd>
                  </div>
                );
              })}
            </dl>
            {result.requires_manual_review && (
              <>
                <h3 className="section-heading">Manual Review Required</h3>
                <pre className="raw-json" tabIndex={0}>
                  {result.manual_review_reasons.join("\n")}
                </pre>
              </>
            )}
            {result.field_audit.length > 0 && (
              <>
                <h3 className="section-heading">Confidence & Audit Trail</h3>
                <dl className="field-grid">
                  {result.field_audit.map((row, idx) => (
                    <div key={`${row.field}-${idx}`} className="field-row">
                      <dt>{row.field}</dt>
                      <dd>
                        {(row.value || "—") +
                          ` (confidence: ${Math.round((row.confidence_score || 0) * 100)}%)` +
                          (row.is_critical ? " [critical]" : "")}
                      </dd>
                    </div>
                  ))}
                </dl>
              </>
            )}
            {result.redacted_image_base64 && (hideSensitive || preview) && (
              <>
                <h3 className="section-heading">{hideSensitive ? "PII Redacted Preview" : "Original Preview"}</h3>
                <img
                  src={hideSensitive ? `data:image/jpeg;base64,${result.redacted_image_base64}` : preview || ""}
                  alt={hideSensitive ? "Redacted preview" : "Original preview"}
                  className="preview-img"
                />
              </>
            )}
            {result.summary && (
              <>
                <h3 className="section-heading">Summary</h3>
                <p className="summary-text">{result.summary}</p>
              </>
            )}
            {result.key_value_pairs.length > 0 && (
              <>
                <h3 className="section-heading">Key-Value Details</h3>
                <dl className="field-grid">
                  {result.key_value_pairs.map((pair, idx) => (
                    <div key={`${pair.key}-${idx}`} className="field-row">
                      <dt>{pair.key}</dt>
                      <dd>{pair.value}</dd>
                    </div>
                  ))}
                </dl>
              </>
            )}
            {result.entities.length > 0 && (
              <>
                <h3 className="section-heading">Entities</h3>
                <dl className="field-grid">
                  {result.entities.map((entity, idx) => (
                    <div key={`${entity.label}-${idx}`} className="field-row">
                      <dt>{entity.label}</dt>
                      <dd>{entity.value}</dd>
                    </div>
                  ))}
                </dl>
              </>
            )}
            {Object.keys(result.normalized_data || {}).length > 0 && (
              <>
                <h3 className="section-heading">Normalized Data</h3>
                <pre className="raw-json" tabIndex={0}>
                  {JSON.stringify(result.normalized_data, null, 2)}
                </pre>
              </>
            )}
            {result.full_text && (
              <>
                <h3 className="section-heading">Full Text</h3>
                <pre className="raw-json" tabIndex={0}>
                  {result.full_text}
                </pre>
              </>
            )}
            <button type="button" className="raw-toggle" onClick={() => setRawOpen((o) => !o)}>
              {rawOpen ? "Hide" : "Show"} raw JSON string
            </button>
            {rawOpen && (
              <pre className="raw-json" tabIndex={0}>
                {result.raw_json_string}
              </pre>
            )}
          </section>
        )}
      </main>

      <footer className="footer">
        <span>API: POST /extract - response is strict JSON schema ready for downstream systems.</span>
      </footer>
    </div>
  );
}
