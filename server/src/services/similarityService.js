/**
 * similarityService.js — Core Plagiarism Detection Algorithms
 *
 * Implements three similarity algorithms:
 * 1. Cosine Similarity — measures angle between TF vectors
 * 2. Token-based (Jaccard) Similarity — set overlap of tokens
 * 3. LCS (Longest Common Subsequence) — structural match
 *
 * Also includes optional Python microservice integration.
 */

const axios = require("axios");
const { tokenize, getTokenFrequency, getUniqueTokens } = require("../utils/tokenizer");

// ============================================================
// 1. COSINE SIMILARITY
// ============================================================

/**
 * Calculate cosine similarity between two token frequency maps.
 *
 * Formula: cos(θ) = (A · B) / (||A|| × ||B||)
 *
 * @param {Map} freqA - Token frequency map for code A
 * @param {Map} freqB - Token frequency map for code B
 * @returns {number} Similarity score (0-100)
 */
function cosineSimilarity(freqA, freqB) {
  // Get all unique tokens from both maps
  const allTokens = new Set([...freqA.keys(), ...freqB.keys()]);

  let dotProduct = 0;
  let magnitudeA = 0;
  let magnitudeB = 0;

  for (const token of allTokens) {
    const a = freqA.get(token) || 0;
    const b = freqB.get(token) || 0;
    dotProduct += a * b;
    magnitudeA += a * a;
    magnitudeB += b * b;
  }

  magnitudeA = Math.sqrt(magnitudeA);
  magnitudeB = Math.sqrt(magnitudeB);

  if (magnitudeA === 0 || magnitudeB === 0) return 0;

  const similarity = (dotProduct / (magnitudeA * magnitudeB)) * 100;
  return Math.round(similarity * 100) / 100; // Round to 2 decimal places
}

// ============================================================
// 2. TOKEN-BASED (JACCARD) SIMILARITY
// ============================================================

/**
 * Calculate Jaccard similarity between two sets of tokens.
 *
 * Formula: J(A, B) = |A ∩ B| / |A ∪ B|
 *
 * @param {Set} setA - Unique tokens from code A
 * @param {Set} setB - Unique tokens from code B
 * @returns {number} Similarity score (0-100)
 */
function tokenSimilarity(setA, setB) {
  if (setA.size === 0 && setB.size === 0) return 0;

  // Count intersection
  let intersection = 0;
  for (const token of setA) {
    if (setB.has(token)) {
      intersection++;
    }
  }

  // Union = |A| + |B| - |A ∩ B|
  const union = setA.size + setB.size - intersection;

  if (union === 0) return 0;

  const similarity = (intersection / union) * 100;
  return Math.round(similarity * 100) / 100;
}

// ============================================================
// 3. LONGEST COMMON SUBSEQUENCE (LCS)
// ============================================================

/**
 * Find the Longest Common Subsequence between two arrays of lines.
 * Returns the length of the LCS and the actual matched lines.
 *
 * Uses dynamic programming with O(n*m) time complexity.
 *
 * @param {string[]} linesA - Lines of code A
 * @param {string[]} linesB - Lines of code B
 * @returns {{ length: number, matches: Array }} LCS result
 */
function lcs(linesA, linesB) {
  const n = linesA.length;
  const m = linesB.length;

  // Build the DP table
  const dp = Array.from({ length: n + 1 }, () => Array(m + 1).fill(0));

  for (let i = 1; i <= n; i++) {
    for (let j = 1; j <= m; j++) {
      if (linesA[i - 1].trim() === linesB[j - 1].trim()) {
        dp[i][j] = dp[i - 1][j - 1] + 1;
      } else {
        dp[i][j] = Math.max(dp[i - 1][j], dp[i][j - 1]);
      }
    }
  }

  // Backtrack to find matched lines
  const matches = [];
  let i = n;
  let j = m;
  while (i > 0 && j > 0) {
    if (linesA[i - 1].trim() === linesB[j - 1].trim()) {
      matches.unshift({
        file1Line: i,
        file2Line: j,
        text: linesA[i - 1].trim(),
      });
      i--;
      j--;
    } else if (dp[i - 1][j] > dp[i][j - 1]) {
      i--;
    } else {
      j--;
    }
  }

  return { length: dp[n][m], matches };
}

/**
 * Calculate LCS similarity as a percentage.
 *
 * @param {string} codeA - Source code A
 * @param {string} codeB - Source code B
 * @returns {{ score: number, matches: Array }} Score and matched sections
 */
function lcsSimilarity(codeA, codeB) {
  const linesA = codeA.split("\n").filter((l) => l.trim().length > 0);
  const linesB = codeB.split("\n").filter((l) => l.trim().length > 0);

  if (linesA.length === 0 && linesB.length === 0) {
    return { score: 0, matches: [] };
  }

  const result = lcs(linesA, linesB);
  const maxLen = Math.max(linesA.length, linesB.length);
  const score = maxLen > 0 ? (result.length / maxLen) * 100 : 0;

  // Group consecutive matches into sections
  const matchSections = groupConsecutiveMatches(result.matches);

  return {
    score: Math.round(score * 100) / 100,
    matches: matchSections,
  };
}

/**
 * Group consecutive matched lines into sections.
 * Instead of listing every individual line, we group them
 * into contiguous blocks for a cleaner result.
 */
function groupConsecutiveMatches(matches) {
  if (matches.length === 0) return [];

  const sections = [];
  let currentSection = {
    file1Start: matches[0].file1Line,
    file1End: matches[0].file1Line,
    file2Start: matches[0].file2Line,
    file2End: matches[0].file2Line,
    matchedText: matches[0].text,
  };

  for (let i = 1; i < matches.length; i++) {
    const match = matches[i];
    // Check if this match is consecutive with the previous one
    if (
      match.file1Line === currentSection.file1End + 1 &&
      match.file2Line === currentSection.file2End + 1
    ) {
      // Extend the current section
      currentSection.file1End = match.file1Line;
      currentSection.file2End = match.file2Line;
      currentSection.matchedText += "\n" + match.text;
    } else {
      // Start a new section
      sections.push({ ...currentSection });
      currentSection = {
        file1Start: match.file1Line,
        file1End: match.file1Line,
        file2Start: match.file2Line,
        file2End: match.file2Line,
        matchedText: match.text,
      };
    }
  }
  sections.push(currentSection);

  return sections;
}

// ============================================================
// 4. PYTHON MICROSERVICE INTEGRATION (Optional)
// ============================================================

/**
 * Call the Python microservice for advanced NLP analysis.
 * Falls back to local Node.js analysis if the service is unavailable.
 *
 * @param {string} codeA - Source code A
 * @param {string} codeB - Source code B
 * @returns {Object|null} Python analysis results, or null if unavailable
 */
async function callPythonService(codeA, codeB) {
  try {
    const url = process.env.PYTHON_SERVICE_URL || "http://localhost:8000";
    const response = await axios.post(
      `${url}/analyze`,
      { code1: codeA, code2: codeB },
      { timeout: 10000 } // 10 second timeout
    );
    return response.data;
  } catch (error) {
    // Python service is optional — log and continue without it
    console.log("⚠️  Python service unavailable, using Node.js algorithms only");
    return null;
  }
}

/**
 * Run all similarity algorithms on two pieces of code.
 *
 * When the Python AI service is available, its code-aware scores
 * (n-gram fingerprinting, pattern analysis) are used as the primary
 * signal. Node.js algorithms serve as a fallback.
 *
 * @param {string} codeA - Preprocessed source code A
 * @param {string} codeB - Preprocessed source code B
 * @returns {Object} Combined similarity results
 */
async function compareTwo(codeA, codeB) {
  // Tokenize both code samples
  const tokensA = tokenize(codeA);
  const tokensB = tokenize(codeB);

  // 1. Cosine Similarity
  const freqA = getTokenFrequency(tokensA);
  const freqB = getTokenFrequency(tokensB);
  const cosineScore = cosineSimilarity(freqA, freqB);

  // 2. Token (Jaccard) Similarity
  const setA = getUniqueTokens(tokensA);
  const setB = getUniqueTokens(tokensB);
  const tokenScore = tokenSimilarity(setA, setB);

  // 3. LCS Similarity
  const lcsResult = lcsSimilarity(codeA, codeB);

  // 4. Try Python microservice for advanced code-aware analysis
  const pythonResult = await callPythonService(codeA, codeB);

  let overall, verdict;

  if (pythonResult && typeof pythonResult.overall === "number") {
    // Python service available — blend its smarter scores in
    // Python weight: 60%, Node weight: 40%
    const nodeScore = cosineScore * 0.4 + tokenScore * 0.25 + lcsResult.score * 0.35;
    overall = pythonResult.overall * 0.6 + nodeScore * 0.4;
    overall = Math.round(overall * 100) / 100;
  } else {
    // Fallback to Node.js only
    overall = cosineScore * 0.4 + tokenScore * 0.25 + lcsResult.score * 0.35;
    overall = Math.round(overall * 100) / 100;
  }

  // Determine verdict
  verdict = "Low";
  if (overall >= 60) verdict = "High";
  else if (overall >= 30) verdict = "Medium";

  return {
    cosine: cosineScore,
    token: tokenScore,
    lcs: lcsResult.score,
    overall,
    verdict,
    matches: lcsResult.matches,
    pythonAnalysis: pythonResult, // null if service unavailable
  };
}

module.exports = {
  cosineSimilarity,
  tokenSimilarity,
  lcsSimilarity,
  callPythonService,
  compareTwo,
};

