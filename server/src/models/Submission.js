/**
 * Submission.js — Mongoose Schema for Code Submissions
 *
 * Stores the original files that users upload or paste.
 * Each submission can contain multiple files for comparison.
 */

const mongoose = require("mongoose");

const fileSchema = new mongoose.Schema({
  name: {
    type: String,
    required: true,
    trim: true,
  },
  content: {
    type: String,
    required: true,
  },
  language: {
    type: String,
    default: "plaintext", // e.g., "javascript", "python", "java"
  },
});

const submissionSchema = new mongoose.Schema(
  {
    // Array of files submitted for comparison
    files: {
      type: [fileSchema],
      validate: {
        validator: (arr) => arr.length >= 2,
        message: "At least 2 files are required for comparison.",
      },
    },
  },
  {
    timestamps: true, // Adds createdAt and updatedAt automatically
  }
);

module.exports = mongoose.model("Submission", submissionSchema);
