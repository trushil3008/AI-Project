/**
 * compareController.js — Request Handlers for Plagiarism Comparison
 *
 * Handles:
 * - POST /api/compare    → Upload files and run comparison
 * - POST /api/detect-ai  → Check if a single file is AI-generated
 * - GET  /api/results    → Get all past results
 * - GET  /api/results/:id → Get a specific result
 */

const axios = require("axios");
const Submission = require("../models/Submission");
const { processSubmission, getAllResults, getResultById } = require("../services/comparisonService");

/**
 * POST /api/compare
 *
 * Accepts files via multipart upload OR JSON body with pasted code.
 * Creates a submission, processes it, and returns the results.
 */
const compareFiles = async (req, res, next) => {
  try {
    let files = [];

    // Option 1: Files uploaded via multer (multipart/form-data)
    if (req.files && req.files.length >= 2) {
      files = req.files.map((f) => ({
        name: f.originalname,
        content: f.buffer.toString("utf-8"),
        language: getLanguageFromFilename(f.originalname),
      }));
    }
    // Option 2: Code pasted as JSON body
    else if (req.body.files && req.body.files.length >= 2) {
      files = req.body.files.map((f) => ({
        name: f.name || "untitled.txt",
        content: f.content,
        language: f.language || "plaintext",
      }));
    }
    // Not enough files provided
    else {
      return res.status(400).json({
        success: false,
        message: "Please provide at least 2 files for comparison.",
      });
    }

    // 1. Save the submission
    const submission = new Submission({ files });
    await submission.save();

    // 2. Process the submission (runs all algorithms)
    const result = await processSubmission(submission._id);

    // 3. Return the result
    res.status(201).json({
      success: true,
      message: "Comparison complete!",
      data: result,
    });
  } catch (error) {
    next(error);
  }
};

/**
 * GET /api/results
 *
 * Returns all past comparison results.
 */
const getResults = async (req, res, next) => {
  try {
    const results = await getAllResults();
    res.status(200).json({
      success: true,
      count: results.length,
      data: results,
    });
  } catch (error) {
    next(error);
  }
};

/**
 * GET /api/results/:id
 *
 * Returns a single comparison result by its ID.
 */
const getResult = async (req, res, next) => {
  try {
    const result = await getResultById(req.params.id);
    if (!result) {
      return res.status(404).json({
        success: false,
        message: "Result not found.",
      });
    }
    res.status(200).json({
      success: true,
      data: result,
    });
  } catch (error) {
    next(error);
  }
};

/**
 * POST /api/detect-ai
 *
 * Accepts a single file (upload or paste) and checks if it was
 * written by AI using the Python microservice's heuristic engine.
 */
const detectAI = async (req, res, next) => {
  try {
    let code = "";
    let filename = "untitled.txt";

    // Option 1: File uploaded via multer
    if (req.files && req.files.length >= 1) {
      code = req.files[0].buffer.toString("utf-8");
      filename = req.files[0].originalname;
    }
    // Option 2: Code pasted as JSON body
    else if (req.body.code) {
      code = req.body.code;
      filename = req.body.name || "untitled.txt";
    } else {
      return res.status(400).json({
        success: false,
        message: "Please provide a file or code to analyze.",
      });
    }

    // Forward to Python microservice
    const pythonUrl = process.env.PYTHON_SERVICE_URL || "http://localhost:8000";
    const response = await axios.post(
      `${pythonUrl}/detect-ai`,
      { code },
      { timeout: 15000 }
    );

    res.status(200).json({
      success: true,
      filename,
      data: response.data,
    });
  } catch (error) {
    // If Python service is down, return a helpful error
    if (error.code === "ECONNREFUSED" || error.code === "ETIMEDOUT") {
      return res.status(503).json({
        success: false,
        message: "AI detection service is unavailable. Please ensure the Python service is running on port 8000.",
      });
    }
    next(error);
  }
};

// ============================================================
// HELPER FUNCTIONS
// ============================================================

/**
 * Detect programming language from file extension.
 */
function getLanguageFromFilename(filename) {
  const ext = filename.split(".").pop().toLowerCase();
  const languageMap = {
    js: "javascript",
    jsx: "javascript",
    ts: "typescript",
    tsx: "typescript",
    py: "python",
    java: "java",
    c: "c",
    cpp: "cpp",
    cs: "csharp",
    rb: "ruby",
    go: "go",
    rs: "rust",
    php: "php",
    html: "html",
    css: "css",
    txt: "plaintext",
  };
  return languageMap[ext] || "plaintext";
}

module.exports = {
  compareFiles,
  getResults,
  getResult,
  detectAI,
};
