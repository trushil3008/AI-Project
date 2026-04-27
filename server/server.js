/**
 * server.js — Entry Point for the Backend
 *
 * Connects to MongoDB and starts the Express server.
 * Environment variables are loaded from .env file.
 */

require("dotenv").config();
const app = require("./src/app");
const connectDB = require("./src/config/db");

const PORT = process.env.PORT || 5000;

// Connect to MongoDB, then start the server
const startServer = async () => {
  await connectDB();

  app.listen(PORT, () => {
    console.log(`\n🚀 Server running on http://localhost:${PORT}`);
    console.log(`📡 API Health: http://localhost:${PORT}/api/health`);
    console.log(`🐍 Python Service: ${process.env.PYTHON_SERVICE_URL || "not configured"}\n`);
  });
};

startServer();
