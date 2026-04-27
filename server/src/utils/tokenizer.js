/**
 * tokenizer.js — Code Tokenization Utility
 *
 * Breaks source code into meaningful tokens for comparison.
 * Tokens include: keywords, operators, identifiers, literals, etc.
 */

/**
 * Tokenize source code into an array of tokens.
 * Splits on whitespace, operators, and punctuation while preserving them.
 *
 * @param {string} code - Preprocessed source code
 * @returns {string[]} Array of tokens
 */
function tokenize(code) {
  if (!code || code.trim().length === 0) {
    return [];
  }

  // Split on word boundaries, operators, and punctuation
  // This regex captures: words, operators, and single characters
  const tokenPattern = /\b\w+\b|[+\-*/%=<>!&|^~?:;,.(){}[\]]/g;
  const tokens = code.match(tokenPattern);

  return tokens || [];
}

/**
 * Create a frequency map (bag of words) from tokens.
 * Used for cosine similarity calculation.
 *
 * @param {string[]} tokens - Array of tokens
 * @returns {Map<string, number>} Token frequency map
 */
function getTokenFrequency(tokens) {
  const freq = new Map();
  for (const token of tokens) {
    const lower = token.toLowerCase();
    freq.set(lower, (freq.get(lower) || 0) + 1);
  }
  return freq;
}

/**
 * Get unique tokens as a Set.
 * Used for token-based (Jaccard) similarity.
 *
 * @param {string[]} tokens - Array of tokens
 * @returns {Set<string>} Set of unique tokens
 */
function getUniqueTokens(tokens) {
  return new Set(tokens.map((t) => t.toLowerCase()));
}

module.exports = {
  tokenize,
  getTokenFrequency,
  getUniqueTokens,
};
