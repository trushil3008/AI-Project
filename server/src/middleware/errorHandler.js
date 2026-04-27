/**
 * errorHandler.js — Global Error Handling Middleware
 *
 * Catches all errors thrown in routes/controllers and returns
 * a clean JSON response to the client.
 */

const errorHandler = (err, req, res, next) => {
  console.error("❌ Error:", err.message);

  // Multer errors (file too large, wrong type, etc.)
  if (err.code === "LIMIT_FILE_SIZE") {
    return res.status(400).json({
      success: false,
      message: "File is too large. Maximum size is 5MB.",
    });
  }

  // Mongoose validation errors
  if (err.name === "ValidationError") {
    const messages = Object.values(err.errors).map((e) => e.message);
    return res.status(400).json({
      success: false,
      message: messages.join(", "),
    });
  }

  // Mongoose bad ObjectId
  if (err.name === "CastError") {
    return res.status(400).json({
      success: false,
      message: "Invalid ID format.",
    });
  }

  // Default server error
  res.status(err.status || 500).json({
    success: false,
    message: err.message || "Internal server error.",
  });
};

module.exports = errorHandler;
