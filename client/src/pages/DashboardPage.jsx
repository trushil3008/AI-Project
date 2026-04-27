/**
 * DashboardPage.jsx — View Past Submissions and Results
 *
 * Lists all past comparison results with:
 * - File names compared
 * - Date submitted
 * - Highest similarity score
 * Click to navigate to the full result view.
 */

import { useEffect, useState } from "react";
import { useNavigate, Link } from "react-router-dom";
import { FiChevronRight, FiClock, FiFile, FiUpload } from "react-icons/fi";
import { getResults } from "../services/api";

export default function DashboardPage() {
  const navigate = useNavigate();
  const [results, setResults] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  useEffect(() => {
    const fetchResults = async () => {
      try {
        const response = await getResults();
        if (response.success) {
          setResults(response.data);
        }
      } catch (err) {
        setError(err.response?.data?.message || "Failed to load results.");
      } finally {
        setLoading(false);
      }
    };

    fetchResults();
  }, []);

  // Get the highest similarity score from comparisons
  const getHighestScore = (comparisons) => {
    if (!comparisons || comparisons.length === 0) return 0;
    return Math.max(...comparisons.map((c) => c.overall));
  };

  // Get the worst verdict from comparisons
  const getWorstVerdict = (comparisons) => {
    if (!comparisons || comparisons.length === 0) return "Low";
    const verdicts = comparisons.map((c) => c.verdict);
    if (verdicts.includes("High")) return "High";
    if (verdicts.includes("Medium")) return "Medium";
    return "Low";
  };

  // Loading State
  if (loading) {
    return (
      <div className="page-container">
        <div className="page-header">
          <h1>Dashboard</h1>
          <p>View all past plagiarism checks.</p>
        </div>
        <div className="loading-container">
          <div className="spinner" />
          <span className="loading-text">Loading submissions...</span>
        </div>
      </div>
    );
  }

  return (
    <div className="page-container">
      {/* Header */}
      <div className="page-header">
        <h1>Dashboard</h1>
        <p>View all past plagiarism checks and their results.</p>
      </div>

      {/* Error */}
      {error && <div className="error-message mb-3">{error}</div>}

      {/* Results List */}
      {results.length > 0 ? (
        <div className="dashboard-grid">
          {results.map((result) => {
            const highestScore = getHighestScore(result.comparisons);
            const worstVerdict = getWorstVerdict(result.comparisons);
            const verdictClass = worstVerdict.toLowerCase();
            const fileCount = result.submission?.files?.length || 0;
            const date = new Date(result.createdAt).toLocaleDateString("en-US", {
              month: "short",
              day: "numeric",
              year: "numeric",
              hour: "2-digit",
              minute: "2-digit",
            });

            return (
              <div
                key={result._id}
                className="dashboard-item"
                onClick={() => navigate(`/results/${result._id}`)}
              >
                <div className="dashboard-item-info">
                  <span className="title">
                    {result.submission?.files
                      ?.map((f) => f.name)
                      .join(", ") || "Unnamed Submission"}
                  </span>
                  <div className="meta">
                    <span style={{ display: "flex", alignItems: "center", gap: 4 }}>
                      <FiFile size={12} /> {fileCount} files
                    </span>
                    <span style={{ display: "flex", alignItems: "center", gap: 4 }}>
                      <FiClock size={12} /> {date}
                    </span>
                  </div>
                </div>

                <div className="dashboard-item-right">
                  <span className={`verdict-badge ${verdictClass}`}>
                    {highestScore}% — {worstVerdict}
                  </span>
                  <FiChevronRight className="arrow" />
                </div>
              </div>
            );
          })}
        </div>
      ) : (
        /* Empty State */
        <div className="empty-state">
          <div className="empty-icon">📋</div>
          <h3>No submissions yet</h3>
          <p>Upload or paste code files to check for plagiarism. Results will appear here.</p>
          <Link to="/" className="btn btn-primary mt-3">
            <FiUpload />
            Upload Files
          </Link>
        </div>
      )}
    </div>
  );
}
