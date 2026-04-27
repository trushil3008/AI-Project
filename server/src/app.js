/**
 * app.js — Express Application Setup
 *
 * Configures middleware, routes, and error handling.
 * This file exports the app without starting the server,
 * so it can be used in testing as well.
 */

const express = require("express");
const cors = require("cors");
const morgan = require("morgan");
const compareRoutes = require("./routes/compareRoutes");
const errorHandler = require("./middleware/errorHandler");

const app = express();

// ============================================================
// MIDDLEWARE
// ============================================================

// Enable CORS for frontend communication
app.use(cors());

// Parse JSON request bodies (for pasted code submissions)
app.use(express.json({ limit: "10mb" }));

// Parse URL-encoded bodies
app.use(express.urlencoded({ extended: true }));

// HTTP request logging (dev format for colored output)
app.use(morgan("dev"));

// ============================================================
// ROUTES
// ============================================================

// Health check endpoint
app.get("/api/health", (req, res) => {
  res.json({ status: "ok", message: "Plagiarism Detector API is running!" });
});

// Main API routes
app.use("/api", compareRoutes);

// ============================================================
// ERROR HANDLING
// ============================================================

// Handle 404 — route not found
app.use((req, res) => {
  res.status(404).json({
    success: false,
    message: `Route ${req.originalUrl} not found.`,
  });
});

// Global error handler (must be last)
app.use(errorHandler);

module.exports = app;
