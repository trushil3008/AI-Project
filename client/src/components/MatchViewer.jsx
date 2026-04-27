/**
 * MatchViewer.jsx — Displays Matched Code Sections
 *
 * Shows the actual code fragments that matched between two files,
 * with line number references.
 */

export default function MatchViewer({ matches }) {
  if (!matches || matches.length === 0) {
    return (
      <div className="match-viewer">
        <p style={{ color: "var(--text-muted)", textAlign: "center", padding: "1rem" }}>
          No matching sections found.
        </p>
      </div>
    );
  }

  return (
    <div className="match-viewer">
      {matches.map((match, index) => (
        <div key={index} className="match-section">
          {/* Header showing line ranges */}
          <div className="match-section-header">
            <span>
              File 1: Lines {match.file1Start}–{match.file1End}
            </span>
            <span>
              File 2: Lines {match.file2Start}–{match.file2End}
            </span>
          </div>

          {/* Matched code content */}
          <div className="match-code">
            <mark>{match.matchedText}</mark>
          </div>
        </div>
      ))}
    </div>
  );
}
