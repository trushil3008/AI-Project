/**
 * preprocessor.js — Code Preprocessing Utilities
 *
 * Cleans and normalizes code before similarity analysis.
 * This step is critical — without preprocessing, minor formatting
 * differences would reduce similarity scores even for copied code.
 */

/**
 * Remove single-line comments from code.
 * Handles: // (JS, Java, C), # (Python, Ruby)
 */
function removeSingleLineComments(code) {
  // Remove // comments (but not inside strings — simplified approach)
  code = code.replace(/\/\/.*$/gm, "");
  // Remove # comments (Python style)
  code = code.replace(/#.*$/gm, "");
  return code;
}

/**
 * Remove multi-line comments from code.
 * Handles: /* ... *\/ (JS, Java, C)
 * Handles: ''' ... ''' and """ ... """ (Python docstrings)
 */
function removeMultiLineComments(code) {
  // Remove /* ... */ comments
  code = code.replace(/\/\*[\s\S]*?\*\//g, "");
  // Remove Python triple-quote strings/docstrings
  code = code.replace(/'''[\s\S]*?'''/g, "");
  code = code.replace(/"""[\s\S]*?"""/g, "");
  return code;
}

/**
 * Normalize whitespace: collapse multiple spaces/tabs/newlines
 * into single spaces, and trim each line.
 */
function normalizeWhitespace(code) {
  return code
    .split("\n") // Split into lines
    .map((line) => line.trim()) // Trim each line
    .filter((line) => line.length > 0) // Remove empty lines
    .join("\n");
}

/**
 * Normalize variable names by replacing common identifier patterns
 * with generic placeholders. This helps detect plagiarism where
 * students just rename variables.
 *
 * Example: "let myVariable = 10;" → "let VAR = 10;"
 */
function normalizeVariableNames(code) {
  // Replace camelCase and snake_case identifiers with a placeholder
  // This is a simplified approach — a full AST parser would be more accurate
  const keywords = new Set([
    "if", "else", "for", "while", "do", "switch", "case", "break",
    "continue", "return", "function", "class", "const", "let", "var",
    "import", "export", "default", "new", "this", "try", "catch",
    "finally", "throw", "async", "await", "yield", "def", "print",
    "int", "float", "string", "boolean", "true", "false", "null",
    "undefined", "void", "public", "private", "protected", "static",
  ]);

  // Replace identifiers that aren't keywords
  return code.replace(/\b([a-zA-Z_]\w*)\b/g, (match) => {
    if (keywords.has(match.toLowerCase())) {
      return match; // Keep keywords as-is
    }
    return "VAR"; // Replace variable/function names
  });
}

/**
 * Full preprocessing pipeline.
 * Applies all cleaning steps in order.
 *
 * @param {string} code - Raw source code
 * @param {boolean} normalizeVars - Whether to normalize variable names
 * @returns {string} Cleaned code ready for analysis
 */
function preprocess(code, normalizeVars = false) {
  let cleaned = code;
  cleaned = removeSingleLineComments(cleaned);
  cleaned = removeMultiLineComments(cleaned);
  cleaned = normalizeWhitespace(cleaned);
  if (normalizeVars) {
    cleaned = normalizeVariableNames(cleaned);
  }
  return cleaned;
}

module.exports = {
  removeSingleLineComments,
  removeMultiLineComments,
  normalizeWhitespace,
  normalizeVariableNames,
  preprocess,
};
