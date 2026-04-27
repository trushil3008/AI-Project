/**
 * Result.js — Mongoose Schema for Comparison Results
 *
 * Stores the output of plagiarism analysis for each submission.
 * Each result contains comparisons for every pair of files.
 */

const mongoose = require("mongoose");

// Schema for a single matched section between two files
const matchSchema = new mongoose.Schema(
  {
    file1Start: Number, // Starting line in file 1
    file1End: Number, // Ending line in file 1
    file2Start: Number, // Starting line in file 2
    file2End: Number, // Ending line in file 2
    matchedText: String, // The actual matched text
  },
  { _id: false }
);

// Schema for a pair-wise comparison result
const comparisonSchema = new mongoose.Schema(
  {
    file1: { type: String, required: true }, // Name of file 1
    file2: { type: String, required: true }, // Name of file 2
    cosine: { type: Number, default: 0 }, // Cosine similarity (0-100)
    token: { type: Number, default: 0 }, // Token-based similarity (0-100)
    lcs: { type: Number, default: 0 }, // LCS similarity (0-100)
    overall: { type: Number, default: 0 }, // Weighted average (0-100)
    verdict: {
      type: String,
      enum: ["Low", "Medium", "High"],
      default: "Low",
    },
    matches: [matchSchema], // Array of matched code sections
  },
  { _id: false }
);

const resultSchema = new mongoose.Schema(
  {
    // Reference to the original submission
    submission: {
      type: mongoose.Schema.Types.ObjectId,
      ref: "Submission",
      required: true,
    },
    // Array of pair-wise comparisons
    comparisons: [comparisonSchema],
  },
  {
    timestamps: true,
  }
);

module.exports = mongoose.model("Result", resultSchema);
