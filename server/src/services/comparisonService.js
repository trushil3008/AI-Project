/**
 * comparisonService.js — Orchestrates the Full Comparison Pipeline
 *
 * Takes a submission with multiple files, runs pairwise comparisons,
 * and stores the results in MongoDB.
 *
 * Pipeline: Files → Preprocess → Compare Pairs → Save Results
 */

const Submission = require("../models/Submission");
const Result = require("../models/Result");
const { preprocess } = require("../utils/preprocessor");
const { compareTwo } = require("./similarityService");

/**
 * Process a submission: compare all file pairs and save results.
 *
 * For N files, we generate N*(N-1)/2 comparisons.
 * Example: 3 files → 3 comparisons (1v2, 1v3, 2v3)
 *
 * @param {string} submissionId - MongoDB ID of the submission
 * @returns {Object} The saved result document
 */
async function processSubmission(submissionId) {
  // 1. Fetch the submission from the database
  const submission = await Submission.findById(submissionId);
  if (!submission) {
    throw new Error("Submission not found");
  }

  const { files } = submission;
  const comparisons = [];

  // 2. Compare every pair of files
  for (let i = 0; i < files.length; i++) {
    for (let j = i + 1; j < files.length; j++) {
      const fileA = files[i];
      const fileB = files[j];

      // Preprocess both files (strip comments, normalize whitespace)
      const cleanA = preprocess(fileA.content, true);
      const cleanB = preprocess(fileB.content, true);

      // Run all similarity algorithms
      const result = await compareTwo(cleanA, cleanB);

      comparisons.push({
        file1: fileA.name,
        file2: fileB.name,
        cosine: result.cosine,
        token: result.token,
        lcs: result.lcs,
        overall: result.overall,
        verdict: result.verdict,
        matches: result.matches,
      });
    }
  }

  // 3. Save the result to the database
  const resultDoc = new Result({
    submission: submissionId,
    comparisons,
  });
  await resultDoc.save();

  return resultDoc;
}

/**
 * Get all results, optionally populated with submission data.
 *
 * @returns {Array} Array of result documents
 */
async function getAllResults() {
  return Result.find()
    .populate("submission", "files createdAt")
    .sort({ createdAt: -1 })
    .lean();
}

/**
 * Get a single result by ID.
 *
 * @param {string} resultId - MongoDB ID of the result
 * @returns {Object} Result document with submission data
 */
async function getResultById(resultId) {
  return Result.findById(resultId)
    .populate("submission", "files createdAt")
    .lean();
}

module.exports = {
  processSubmission,
  getAllResults,
  getResultById,
};
