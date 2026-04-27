/**
 * AIDetectPage.jsx — AI Code Detection Page
 *
 * Upload or paste a SINGLE code file to check if it was
 * written by AI (ChatGPT, Copilot, etc.).
 *
 * Displays:
 * - AI probability percentage with animated gauge
 * - Verdict (Human-Written → Highly Likely AI)
 * - Individual heuristic indicators with severity badges
 * - Code statistics summary
 */

import { useState, useCallback } from "react";
import { useDropzone } from "react-dropzone";
import {
  FiUploadCloud, FiCpu, FiAlertCircle, FiCheckCircle,
  FiAlertTriangle, FiInfo, FiFile, FiX, FiCode,
} from "react-icons/fi";
import { detectAIByUpload, detectAIByPaste } from "../services/api";

export default function AIDetectPage() {
  const [mode, setMode] = useState("paste"); // "upload" or "paste"
  const [file, setFile] = useState(null);
  const [pastedCode, setPastedCode] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [result, setResult] = useState(null);

  // Dropzone for single file
  const onDrop = useCallback((acceptedFiles) => {
    if (acceptedFiles.length > 0) {
      setFile(acceptedFiles[0]);
    }
  }, []);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      "text/*": [".js", ".jsx", ".ts", ".tsx", ".py", ".java", ".c", ".cpp", ".txt", ".md"],
    },
    multiple: false,
    maxFiles: 1,
  });

  // Submit handler
  const handleAnalyze = async () => {
    setError("");
    setLoading(true);
    setResult(null);

    try {
      let response;

      if (mode === "upload") {
        if (!file) {
          throw new Error("Please upload a file to analyze.");
        }
        response = await detectAIByUpload(file);
      } else {
        if (!pastedCode.trim()) {
          throw new Error("Please paste some code to analyze.");
        }
        response = await detectAIByPaste(pastedCode);
      }

      if (response.success) {
        setResult(response.data);
      } else {
        throw new Error(response.message || "Analysis failed.");
      }
    } catch (err) {
      setError(err.response?.data?.message || err.message || "Something went wrong.");
    } finally {
      setLoading(false);
    }
  };

  const canSubmit = mode === "upload" ? !!file : pastedCode.trim().length > 0;

  // Get color based on AI probability
  const getAIColor = (probability) => {
    if (probability >= 75) return "var(--verdict-high)";
    if (probability >= 50) return "var(--verdict-medium)";
    if (probability >= 30) return "#f59e0b";
    return "var(--verdict-low)";
  };

  const getAIBgColor = (probability) => {
    if (probability >= 75) return "var(--verdict-high-bg)";
    if (probability >= 50) return "var(--verdict-medium-bg)";
    if (probability >= 30) return "rgba(245, 158, 11, 0.12)";
    return "var(--verdict-low-bg)";
  };

  const getSeverityIcon = (severity) => {
    switch (severity) {
      case "high": return <FiAlertCircle style={{ color: "var(--verdict-high)" }} />;
      case "medium": return <FiAlertTriangle style={{ color: "var(--verdict-medium)" }} />;
      default: return <FiInfo style={{ color: "var(--text-muted)" }} />;
    }
  };

  return (
    <div className="page-container">
      {/* Header */}
      <div className="page-header">
        <h1>AI Code Detector</h1>
        <p>Upload or paste a single code file to check if it was written by AI.</p>
      </div>

      {/* Mode Tabs */}
      <div className="tabs">
        <button
          className={`tab ${mode === "paste" ? "active" : ""}`}
          onClick={() => setMode("paste")}
        >
          <FiCode style={{ marginRight: 6 }} />
          Paste Code
        </button>
        <button
          className={`tab ${mode === "upload" ? "active" : ""}`}
          onClick={() => setMode("upload")}
        >
          <FiUploadCloud style={{ marginRight: 6 }} />
          Upload File
        </button>
      </div>

      {/* Error */}
      {error && (
        <div className="error-message mb-3">
          <FiAlertCircle size={18} />
          {error}
        </div>
      )}

      {/* Input Section */}
      {!result && (
        <>
          {mode === "upload" ? (
            <div>
              <div
                {...getRootProps()}
                className={`dropzone ${isDragActive ? "active" : ""}`}
              >
                <input {...getInputProps()} />
                <div className="dropzone-icon"><FiUploadCloud /></div>
                <h3>{isDragActive ? "Drop your file here..." : "Drag & drop a code file here"}</h3>
                <p>or click to browse (single file only)</p>
              </div>
              {file && (
                <div className="file-list">
                  <div className="file-chip">
                    <FiFile className="file-icon" />
                    <span>{file.name}</span>
                    <span className="remove" onClick={() => setFile(null)}><FiX /></span>
                  </div>
                </div>
              )}
            </div>
          ) : (
            <div className="card code-editor-group">
              <div className="code-editor">
                <textarea
                  placeholder="Paste your code here to check if it was written by AI..."
                  value={pastedCode}
                  onChange={(e) => setPastedCode(e.target.value)}
                  spellCheck={false}
                  style={{ minHeight: "280px" }}
                />
              </div>
            </div>
          )}

          {/* Analyze Button */}
          <div className="mt-4" style={{ textAlign: "center" }}>
            <button
              className="btn btn-primary"
              onClick={handleAnalyze}
              disabled={!canSubmit || loading}
              style={{ padding: "0.8rem 3rem", fontSize: "1rem" }}
            >
              {loading ? (
                <>
                  <span className="spinner" style={{ width: 18, height: 18, borderWidth: 2 }} />
                  Analyzing...
                </>
              ) : (
                <>
                  <FiCpu />
                  Detect AI Code
                </>
              )}
            </button>
          </div>
        </>
      )}

      {/* Results Section */}
      {result && (
        <div className="flex flex-col gap-2">
          {/* AI Probability Gauge */}
          <div className="card" style={{ textAlign: "center", padding: "2.5rem" }}>
            {/* Large circular gauge */}
            <div
              className="overall-score"
              style={{
                width: "140px",
                height: "140px",
                borderColor: getAIColor(result.ai_probability),
                margin: "0 auto 1.5rem",
                borderWidth: "4px",
              }}
            >
              <span
                className="score-number"
                style={{
                  fontSize: "2rem",
                  color: getAIColor(result.ai_probability),
                }}
              >
                {result.ai_probability}%
              </span>
              <span className="score-subtitle">AI Probability</span>
            </div>

            {/* Verdict Badge */}
            <div
              style={{
                display: "inline-block",
                padding: "0.5rem 1.5rem",
                borderRadius: "100px",
                fontWeight: 700,
                fontSize: "0.9rem",
                color: getAIColor(result.ai_probability),
                background: getAIBgColor(result.ai_probability),
                letterSpacing: "0.5px",
              }}
            >
              {result.verdict}
            </div>

            {/* Summary Stats */}
            {result.summary && (
              <div
                style={{
                  display: "flex",
                  justifyContent: "center",
                  gap: "2rem",
                  marginTop: "1.5rem",
                  color: "var(--text-muted)",
                  fontSize: "0.8rem",
                }}
              >
                <span>{result.summary.total_lines} total lines</span>
                <span>{result.summary.code_lines} code lines</span>
                <span>{result.summary.comment_lines} comments</span>
                <span>{result.summary.functions_found} functions</span>
              </div>
            )}
          </div>

          {/* Heuristic Score Breakdown */}
          {result.details && (
            <div className="card">
              <h3 style={{ marginBottom: "1rem", fontSize: "1rem", fontWeight: 600 }}>
                Heuristic Breakdown
              </h3>
              <div className="score-bars">
                {Object.entries(result.details)
                  .sort((a, b) => b[1] - a[1])
                  .map(([key, score]) => {
                    const label = key
                      .replace(/_/g, " ")
                      .replace(/\b\w/g, (c) => c.toUpperCase());
                    const getBarClass = (s) => {
                      if (s >= 60) return "high";
                      if (s >= 30) return "medium";
                      return "low";
                    };
                    return (
                      <div className="score-row" key={key}>
                        <span className="score-label" style={{ width: "160px" }}>{label}</span>
                        <div className="score-bar-track">
                          <div
                            className={`score-bar-fill ${getBarClass(score)}`}
                            style={{ width: `${score}%` }}
                          />
                        </div>
                        <span className="score-value">{score}%</span>
                      </div>
                    );
                  })}
              </div>
            </div>
          )}

          {/* Indicators List */}
          {result.indicators && result.indicators.length > 0 && (
            <div className="card">
              <h3 style={{ marginBottom: "1rem", fontSize: "1rem", fontWeight: 600 }}>
                AI Indicators Found ({result.indicators.length})
              </h3>
              <div className="flex flex-col gap-1">
                {result.indicators.map((indicator, index) => (
                  <div
                    key={index}
                    style={{
                      display: "flex",
                      gap: "0.75rem",
                      padding: "0.75rem 1rem",
                      background: "var(--bg-input)",
                      borderRadius: "var(--radius-sm)",
                      border: "1px solid var(--border-color)",
                      alignItems: "flex-start",
                    }}
                  >
                    <div style={{ marginTop: "2px", flexShrink: 0 }}>
                      {getSeverityIcon(indicator.severity)}
                    </div>
                    <div style={{ flex: 1 }}>
                      <div style={{
                        fontWeight: 600,
                        fontSize: "0.85rem",
                        marginBottom: "0.25rem",
                        display: "flex",
                        justifyContent: "space-between",
                        alignItems: "center",
                      }}>
                        <span>{indicator.pattern}</span>
                        <span
                          style={{
                            fontSize: "0.7rem",
                            padding: "2px 8px",
                            borderRadius: "100px",
                            fontWeight: 700,
                            background:
                              indicator.severity === "high"
                                ? "var(--verdict-high-bg)"
                                : indicator.severity === "medium"
                                ? "var(--verdict-medium-bg)"
                                : "var(--bg-card)",
                            color:
                              indicator.severity === "high"
                                ? "var(--verdict-high)"
                                : indicator.severity === "medium"
                                ? "var(--verdict-medium)"
                                : "var(--text-muted)",
                          }}
                        >
                          {indicator.severity.toUpperCase()}
                        </span>
                      </div>
                      <div style={{
                        color: "var(--text-muted)",
                        fontSize: "0.8rem",
                        lineHeight: "1.5",
                      }}>
                        {indicator.description}
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* No indicators */}
          {result.indicators && result.indicators.length === 0 && (
            <div className="card" style={{ textAlign: "center", padding: "2rem" }}>
              <FiCheckCircle size={32} style={{ color: "var(--verdict-low)", marginBottom: "0.75rem" }} />
              <h3 style={{ color: "var(--verdict-low)", marginBottom: "0.25rem" }}>No AI Indicators Found</h3>
              <p style={{ color: "var(--text-muted)", fontSize: "0.85rem" }}>
                This code does not show common patterns associated with AI-generated code.
              </p>
            </div>
          )}

          {/* Analyze Another */}
          <div className="mt-2" style={{ textAlign: "center" }}>
            <button
              className="btn btn-secondary"
              onClick={() => {
                setResult(null);
                setPastedCode("");
                setFile(null);
                setError("");
              }}
            >
              Analyze Another File
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
