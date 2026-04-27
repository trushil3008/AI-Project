/**
 * SimilarityCard.jsx — Displays Comparison Results for a File Pair
 *
 * Shows:
 * - File names being compared
 * - Verdict badge (Low/Medium/High)
 * - Individual algorithm scores with animated bars
 * - Overall score circle
 */

import { useState } from "react";
import { FiChevronDown, FiChevronUp } from "react-icons/fi";
import MatchViewer from "./MatchViewer";

export default function SimilarityCard({ comparison }) {
  const [expanded, setExpanded] = useState(false);

  // Get color class based on score
  const getColorClass = (score) => {
    if (score >= 60) return "high";
    if (score >= 30) return "medium";
    return "low";
  };

  const verdictClass = comparison.verdict.toLowerCase();

  return (
    <div className="similarity-card">
      {/* Header: File names and verdict */}
      <div className="similarity-header">
        <div className="similarity-files">
          <span>{comparison.file1}</span>
          <span className="vs">VS</span>
          <span>{comparison.file2}</span>
        </div>
        <span className={`verdict-badge ${verdictClass}`}>
          {comparison.verdict} Similarity
        </span>
      </div>

      {/* Overall Score Circle */}
      <div className={`overall-score ${verdictClass}`}>
        <span className="score-number">{comparison.overall}%</span>
        <span className="score-subtitle">Overall</span>
      </div>

      {/* Individual Score Bars */}
      <div className="score-bars">
        {/* Cosine Similarity */}
        <div className="score-row">
          <span className="score-label">Cosine</span>
          <div className="score-bar-track">
            <div
              className={`score-bar-fill ${getColorClass(comparison.cosine)}`}
              style={{ width: `${comparison.cosine}%` }}
            />
          </div>
          <span className="score-value">{comparison.cosine}%</span>
        </div>

        {/* Token Similarity */}
        <div className="score-row">
          <span className="score-label">Token Match</span>
          <div className="score-bar-track">
            <div
              className={`score-bar-fill ${getColorClass(comparison.token)}`}
              style={{ width: `${comparison.token}%` }}
            />
          </div>
          <span className="score-value">{comparison.token}%</span>
        </div>

        {/* LCS Similarity */}
        <div className="score-row">
          <span className="score-label">LCS</span>
          <div className="score-bar-track">
            <div
              className={`score-bar-fill ${getColorClass(comparison.lcs)}`}
              style={{ width: `${comparison.lcs}%` }}
            />
          </div>
          <span className="score-value">{comparison.lcs}%</span>
        </div>
      </div>

      {/* Expandable Match Viewer */}
      {comparison.matches && comparison.matches.length > 0 && (
        <div className="mt-3">
          <button
            className="btn btn-ghost w-full"
            onClick={() => setExpanded(!expanded)}
          >
            {expanded ? <FiChevronUp /> : <FiChevronDown />}
            {expanded ? "Hide" : "Show"} Matched Sections ({comparison.matches.length})
          </button>

          {expanded && <MatchViewer matches={comparison.matches} />}
        </div>
      )}
    </div>
  );
}
