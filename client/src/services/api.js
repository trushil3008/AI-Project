/**
 * api.js — API Service Layer
 *
 * Centralizes all HTTP communication with the backend.
 * Uses Axios for cleaner request/response handling.
 */

import axios from "axios";

// Create an Axios instance with default settings
const api = axios.create({
  baseURL: "/api", // Vite proxy will forward to backend
  headers: {
    "Content-Type": "application/json",
  },
});

/**
 * Compare files by uploading them via multipart form data.
 *
 * @param {File[]} files - Array of File objects from input/dropzone
 * @returns {Object} Comparison result
 */
export const compareByUpload = async (files) => {
  const formData = new FormData();
  files.forEach((file) => {
    formData.append("files", file);
  });

  const response = await api.post("/compare", formData, {
    headers: { "Content-Type": "multipart/form-data" },
  });
  return response.data;
};

/**
 * Compare files by sending pasted code as JSON.
 *
 * @param {Array<{name: string, content: string}>} files - Array of code objects
 * @returns {Object} Comparison result
 */
export const compareByPaste = async (files) => {
  const response = await api.post("/compare", { files });
  return response.data;
};

/**
 * Get all past comparison results.
 *
 * @returns {Object} Array of results
 */
export const getResults = async () => {
  const response = await api.get("/results");
  return response.data;
};

/**
 * Get a specific result by ID.
 *
 * @param {string} id - Result ID
 * @returns {Object} Result data
 */
export const getResultById = async (id) => {
  const response = await api.get(`/results/${id}`);
  return response.data;
};

/**
 * Detect if a single uploaded file is AI-generated.
 *
 * @param {File} file - A single File object
 * @returns {Object} AI detection result
 */
export const detectAIByUpload = async (file) => {
  const formData = new FormData();
  formData.append("files", file);

  const response = await api.post("/detect-ai", formData, {
    headers: { "Content-Type": "multipart/form-data" },
  });
  return response.data;
};

/**
 * Detect if pasted code is AI-generated.
 *
 * @param {string} code - The code to analyze
 * @param {string} name - Optional filename
 * @returns {Object} AI detection result
 */
export const detectAIByPaste = async (code, name = "untitled.txt") => {
  const response = await api.post("/detect-ai", { code, name });
  return response.data;
};

export default api;
