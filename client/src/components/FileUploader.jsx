/**
 * FileUploader.jsx — Drag & Drop File Upload Component
 *
 * Uses react-dropzone for drag-and-drop functionality.
 * Shows uploaded files as removable chips.
 */

import { useCallback } from "react";
import { useDropzone } from "react-dropzone";
import { FiUploadCloud, FiFile, FiX } from "react-icons/fi";

export default function FileUploader({ files, onFilesChange }) {
  // Handle dropped/selected files
  const onDrop = useCallback(
    (acceptedFiles) => {
      // Add new files to existing list (avoid duplicates by name)
      const existingNames = new Set(files.map((f) => f.name));
      const newFiles = acceptedFiles.filter((f) => !existingNames.has(f.name));
      onFilesChange([...files, ...newFiles]);
    },
    [files, onFilesChange]
  );

  // Remove a file from the list
  const removeFile = (index) => {
    onFilesChange(files.filter((_, i) => i !== index));
  };

  // react-dropzone configuration
  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      "text/*": [".js", ".jsx", ".ts", ".tsx", ".py", ".java", ".c", ".cpp", ".txt", ".md"],
    },
    multiple: true,
  });

  return (
    <div>
      {/* Dropzone Area */}
      <div
        {...getRootProps()}
        className={`dropzone ${isDragActive ? "active" : ""}`}
      >
        <input {...getInputProps()} />
        <div className="dropzone-icon">
          <FiUploadCloud />
        </div>
        <h3>
          {isDragActive
            ? "Drop your files here..."
            : "Drag & drop code files here"}
        </h3>
        <p>or click to browse • Supports .js, .py, .java, .cpp, .txt and more</p>
      </div>

      {/* Uploaded Files List */}
      {files.length > 0 && (
        <div className="file-list">
          {files.map((file, index) => (
            <div key={index} className="file-chip">
              <FiFile className="file-icon" />
              <span>{file.name}</span>
              <span className="remove" onClick={() => removeFile(index)}>
                <FiX />
              </span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
