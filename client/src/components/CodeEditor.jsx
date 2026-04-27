/**
 * CodeEditor.jsx — Code Paste Text Area Component
 *
 * Allows users to paste code directly with a filename input.
 * Used as an alternative to file upload.
 */

import { FiX, FiCode } from "react-icons/fi";

export default function CodeEditor({ index, data, onChange, onRemove, canRemove }) {
  return (
    <div className="card code-editor-group">
      {/* Header with filename input and remove button */}
      <div className="code-editor-header">
        <div className="flex items-center gap-1">
          <FiCode style={{ color: "var(--accent-primary)" }} />
          <input
            type="text"
            placeholder={`file${index + 1}.txt`}
            value={data.name}
            onChange={(e) => onChange(index, "name", e.target.value)}
          />
        </div>
        {canRemove && (
          <button className="remove-btn" onClick={() => onRemove(index)} title="Remove this file">
            <FiX />
          </button>
        )}
      </div>

      {/* Code Text Area */}
      <div className="code-editor">
        <textarea
          placeholder="Paste your code here..."
          value={data.content}
          onChange={(e) => onChange(index, "content", e.target.value)}
          spellCheck={false}
        />
      </div>
    </div>
  );
}
