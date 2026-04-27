/**
 * upload.js — Multer Middleware Configuration
 *
 * Configures file upload handling:
 * - Uses memory storage (files stored in buffer, not disk)
 * - Limits file size to 5MB
 * - Accepts common code file types
 */

const multer = require("multer");

// Use memory storage — files are stored as Buffer objects in req.files
// This is simpler than disk storage and works well for text files
const storage = multer.memoryStorage();

// File filter — only accept code/text files
const fileFilter = (req, file, cb) => {
  const allowedExtensions = [
    ".js", ".jsx", ".ts", ".tsx", ".py", ".java",
    ".c", ".cpp", ".h", ".cs", ".rb", ".go", ".rs",
    ".php", ".html", ".css", ".txt", ".md",
  ];

  const ext = "." + file.originalname.split(".").pop().toLowerCase();

  if (allowedExtensions.includes(ext)) {
    cb(null, true); // Accept the file
  } else {
    cb(new Error(`File type ${ext} is not allowed. Please upload code or text files.`), false);
  }
};

const upload = multer({
  storage,
  fileFilter,
  limits: {
    fileSize: 5 * 1024 * 1024, // 5MB max per file
  },
});

module.exports = upload;
