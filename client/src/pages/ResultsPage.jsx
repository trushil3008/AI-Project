/**
 * ResultsPage.jsx — Displays Comparison Results for a Submission
 *
 * Shows all pairwise comparisons with:
 * - Similarity scores (cosine, token, LCS)
 * - Verdict badges
 * - Expandable matched code sections
 */

import { useEffect, useState } from "react";
import { useParams, Link } from "react-router-dom";
import { FiArrowLeft, FiClock } from "react-icons/fi";
import { getResultById } from "../services/api";
import SimilarityCard from "../components/SimilarityCard";

export default function ResultsPage() {
  const { id } = useParams();
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  useEffect(() => {
    const fetchResult = async () => {
      try {
        const response = await getResultById(id);
        if (response.success) {
          setResult(response.data);
        } else {
          setError("Result not found.");
        }
      } catch (err) {
        setError(err.response?.data?.message || "Failed to load results.");
      } finally {
        setLoading(false);
      }
    };

    fetchResult();
  }, [id]);

  // Loading State
  if (loading) {
    return (
      <div className="page-container">
        <div className="loading-container">
          <div className="spinner" />
          <span className="loading-text">Loading results...</span>
        </div>
      </div>
    );
  }

  // Error State
  if (error) {
    return (
      <div className="page-container">
        <div className="error-message">{error}</div>
        <Link to="/" className="btn btn-secondary mt-3">
          <FiArrowLeft /> Back to Upload
        </Link>
      </div>
    );
  }

  // Format date
  const createdAt = result?.createdAt
    ? new Date(result.createdAt).toLocaleString()
    : "Unknown";

  return (
    <div className="page-container">
      {/* Header */}
      <div className="page-header">
        <Link to="/dashboard" className="btn btn-ghost" style={{ marginBottom: "0.5rem" }}>
          <FiArrowLeft /> Back to Dashboard
        </Link>
        <h1>Analysis Results</h1>
        <p style={{ display: "flex", alignItems: "center", gap: "0.5rem" }}>
          <FiClock size={14} />
          {createdAt}
          {" • "}
          {result?.comparisons?.length || 0} comparison(s)
        </p>
      </div>

      {/* Comparison Cards */}
      <div className="flex flex-col gap-2">
        {result?.comparisons?.map((comparison, index) => (
          <SimilarityCard key={index} comparison={comparison} />
        ))}
      </div>

      {/* No comparisons */}
      {(!result?.comparisons || result.comparisons.length === 0) && (
        <div className="empty-state">
          <div className="empty-icon">📊</div>
          <h3>No comparisons available</h3>
          <p>Something went wrong during analysis.</p>
        </div>
      )}
    </div>
  );
}
