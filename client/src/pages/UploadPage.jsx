/**
 * UploadPage.jsx — Main Upload & Compare Page
 *
 * Two modes:
 * 1. Upload — Drag & drop code files
 * 2. Paste — Manually type/paste code into text areas
 *
 * On submit, sends to backend and navigates to results.
 */

import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { FiUploadCloud, FiEdit3, FiPlus, FiZap, FiAlertCircle } from "react-icons/fi";
import FileUploader from "../components/FileUploader";
import CodeEditor from "../components/CodeEditor";
import { compareByUpload, compareByPaste } from "../services/api";

export default function UploadPage() {
  const navigate = useNavigate();

  // Current mode: "upload" or "paste"
  const [mode, setMode] = useState("upload");

  // Upload mode state
  const [uploadedFiles, setUploadedFiles] = useState([]);

  // Paste mode state — start with 2 empty editors
  const [pastedFiles, setPastedFiles] = useState([
    { name: "file1.txt", content: "" },
    { name: "file2.txt", content: "" },
  ]);

  // Submission state
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  // ============================================================
  // PASTE MODE HANDLERS
  // ============================================================

  const updatePastedFile = (index, field, value) => {
    const updated = [...pastedFiles];
    updated[index] = { ...updated[index], [field]: value };
    setPastedFiles(updated);
  };

  const addPastedFile = () => {
    setPastedFiles([
      ...pastedFiles,
      { name: `file${pastedFiles.length + 1}.txt`, content: "" },
    ]);
  };

  const removePastedFile = (index) => {
    if (pastedFiles.length <= 2) return; // Minimum 2 files
    setPastedFiles(pastedFiles.filter((_, i) => i !== index));
  };

  // ============================================================
  // SUBMIT HANDLER
  // ============================================================

  const handleSubmit = async () => {
    setError("");
    setLoading(true);

    try {
      let result;

      if (mode === "upload") {
        // Validate uploaded files
        if (uploadedFiles.length < 2) {
          throw new Error("Please upload at least 2 files for comparison.");
        }
        result = await compareByUpload(uploadedFiles);
      } else {
        // Validate pasted code
        const nonEmpty = pastedFiles.filter((f) => f.content.trim().length > 0);
        if (nonEmpty.length < 2) {
          throw new Error("Please paste code in at least 2 editors.");
        }
        result = await compareByPaste(nonEmpty);
      }

      // Navigate to results page with the result ID
      if (result.success && result.data) {
        navigate(`/results/${result.data._id}`);
      }
    } catch (err) {
      setError(err.response?.data?.message || err.message || "Something went wrong.");
    } finally {
      setLoading(false);
    }
  };

  // Check if submit is possible
  const canSubmit =
    mode === "upload"
      ? uploadedFiles.length >= 2
      : pastedFiles.filter((f) => f.content.trim().length > 0).length >= 2;

  return (
    <div className="page-container">
      {/* Page Header */}
      <div className="page-header">
        <h1>Check for Plagiarism</h1>
        <p>Upload or paste code files to detect similarity using AI-powered algorithms.</p>
      </div>

      {/* Mode Tabs */}
      <div className="tabs">
        <button
          className={`tab ${mode === "upload" ? "active" : ""}`}
          onClick={() => setMode("upload")}
        >
          <FiUploadCloud style={{ marginRight: 6 }} />
          Upload Files
        </button>
        <button
          className={`tab ${mode === "paste" ? "active" : ""}`}
          onClick={() => setMode("paste")}
        >
          <FiEdit3 style={{ marginRight: 6 }} />
          Paste Code
        </button>
      </div>

      {/* Error Message */}
      {error && (
        <div className="error-message mb-3">
          <FiAlertCircle size={18} />
          {error}
        </div>
      )}

      {/* Upload Mode */}
      {mode === "upload" && (
        <FileUploader files={uploadedFiles} onFilesChange={setUploadedFiles} />
      )}

      {/* Paste Mode */}
      {mode === "paste" && (
        <div className="flex flex-col gap-2">
          {pastedFiles.map((file, index) => (
            <CodeEditor
              key={index}
              index={index}
              data={file}
              onChange={updatePastedFile}
              onRemove={removePastedFile}
              canRemove={pastedFiles.length > 2}
            />
          ))}

          {/* Add another editor */}
          <button className="btn btn-secondary w-full" onClick={addPastedFile}>
            <FiPlus />
            Add Another File
          </button>
        </div>
      )}

      {/* Submit Button */}
      <div className="mt-4" style={{ textAlign: "center" }}>
        <button
          className="btn btn-primary"
          onClick={handleSubmit}
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
              <FiZap />
              Compare Files
            </>
          )}
        </button>
      </div>
    </div>
  );
}
