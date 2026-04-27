/**
 * compareRoutes.js — Express Routes for Comparison API
 *
 * Defines the REST endpoints and connects them to controllers.
 */

const express = require("express");
const router = express.Router();
const { compareFiles, getResults, getResult, detectAI } = require("../controllers/compareController");
const upload = require("../middleware/upload");

// POST /api/compare — Upload and compare files
// 'files' is the field name in the multipart form
// max 10 files per submission
router.post("/compare", upload.array("files", 10), compareFiles);

// POST /api/detect-ai — Check if a single file is AI-generated
router.post("/detect-ai", upload.array("files", 1), detectAI);

// GET /api/results — Get all past results
router.get("/results", getResults);

// GET /api/results/:id — Get a specific result
router.get("/results/:id", getResult);

module.exports = router;

