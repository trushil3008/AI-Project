"""
app.py — Advanced AI Code Plagiarism Detection Microservice

This service performs INTELLIGENT code similarity analysis, not just
text matching. It understands code structure and ignores noise like
braces, semicolons, and boilerplate syntax.

Algorithms:
  1. Structural Cosine Similarity — TF-IDF on meaningful code tokens only
  2. N-gram Fingerprinting       — Winnowing algorithm for structural matching
  3. AST-aware Sequence Matching  — Logic-level diff, not character-level
  4. Semantic Token Similarity    — Normalized identifier comparison
    5. Code Embedding Similarity    - Transformer embeddings (CodeBERT)
    6. Optional ML Calibrator       - Trained logistic model over similarity scores

Key design decisions:
    - Transformer embeddings for semantic similarity
    - Optional ML calibrator can be trained locally with labeled pairs
    - Code-aware tokenizer strips syntactic noise BEFORE comparison
    - Variable/function names are normalized so renaming doesn't fool it
    - Common language boilerplate is filtered out

Run: python app.py
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import re
import os
import json
import math
import hashlib
import difflib
import zlib
from collections import Counter, OrderedDict

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

app = Flask(__name__)
CORS(app)


# ============================================================
# 1. ADVANCED CODE PREPROCESSING
# ============================================================

# Keywords that appear in EVERY program — low similarity signal
LANGUAGE_KEYWORDS = {
    # JavaScript / TypeScript
    "var",
    "let",
    "const",
    "function",
    "return",
    "if",
    "else",
    "for",
    "while",
    "do",
    "switch",
    "case",
    "break",
    "continue",
    "new",
    "this",
    "class",
    "extends",
    "import",
    "export",
    "default",
    "from",
    "try",
    "catch",
    "finally",
    "throw",
    "async",
    "await",
    "yield",
    "typeof",
    "instanceof",
    "void",
    "delete",
    "in",
    "of",
    "true",
    "false",
    "null",
    "undefined",
    "console",
    "log",
    # Python
    "def",
    "class",
    "return",
    "if",
    "elif",
    "else",
    "for",
    "while",
    "import",
    "from",
    "as",
    "try",
    "except",
    "finally",
    "raise",
    "with",
    "pass",
    "lambda",
    "yield",
    "global",
    "nonlocal",
    "assert",
    "True",
    "False",
    "None",
    "and",
    "or",
    "not",
    "is",
    "in",
    "print",
    "self",
    "range",
    "len",
    "list",
    "dict",
    "set",
    "tuple",
    "int",
    "str",
    "float",
    # Java / C / C++
    "public",
    "private",
    "protected",
    "static",
    "final",
    "abstract",
    "void",
    "int",
    "float",
    "double",
    "char",
    "boolean",
    "long",
    "short",
    "byte",
    "String",
    "class",
    "interface",
    "extends",
    "implements",
    "package",
    "import",
    "new",
    "this",
    "super",
    "return",
    "if",
    "else",
    "for",
    "while",
    "do",
    "switch",
    "case",
    "break",
    "continue",
    "try",
    "catch",
    "finally",
    "throw",
    "throws",
    "main",
    "System",
    "out",
    "println",
    "printf",
    "scanf",
    "stdio",
    "stdlib",
    "include",
    "using",
    "namespace",
    "std",
    "cout",
    "cin",
    "endl",
}

# Syntax characters that are NOISE — they appear in ALL code
SYNTAX_NOISE = set("{}()[];,.:")

# Embedding model configuration (overridable via env vars)
EMBED_MODEL_NAME = os.getenv("EMBED_MODEL_NAME", "microsoft/codebert-base")
EMBED_MAX_LENGTH = int(os.getenv("EMBED_MAX_LENGTH", "512"))
EMBED_CHUNK_CHARS = int(os.getenv("EMBED_CHUNK_CHARS", "2200"))
EMBED_CHUNK_OVERLAP = int(os.getenv("EMBED_CHUNK_OVERLAP", "200"))
EMBED_CACHE_SIZE = int(os.getenv("EMBED_CACHE_SIZE", "128"))
EMBED_DEVICE = os.getenv(
    "EMBED_DEVICE", "cuda" if torch.cuda.is_available() else "cpu"
)

PLAGIARISM_MODEL_PATH = os.getenv(
    "PLAGIARISM_MODEL_PATH",
    os.path.join(os.path.dirname(__file__), "plagiarism_model.json"),
)

_EMBED_TOKENIZER = None
_EMBED_MODEL = None
_EMBED_CACHE = OrderedDict()

PLAGIARISM_MODEL = None
PLAGIARISM_MODEL_META = {
    "version": 1,
    "features": [
        "cosine_similarity",
        "ngram_fingerprint",
        "structural_similarity",
        "pattern_similarity",
        "embedding_similarity",
    ],
}


def strip_comments(code):
    """Remove all comments from code (single-line, multi-line, docstrings)."""
    # Remove multi-line comments /* ... */
    code = re.sub(r"/\*[\s\S]*?\*/", "", code)
    # Remove Python docstrings """ ... """ and ''' ... '''
    code = re.sub(r'"""[\s\S]*?"""', "", code)
    code = re.sub(r"'''[\s\S]*?'''", "", code)
    # Remove single-line comments // and #
    code = re.sub(r"//.*$", "", code, flags=re.MULTILINE)
    code = re.sub(r"#.*$", "", code, flags=re.MULTILINE)
    return code


def strip_string_literals(code):
    """Replace string literals with a placeholder token.
    "hello world" → STR_LITERAL
    This prevents string content from inflating similarity scores."""
    code = re.sub(r'"(?:[^"\\]|\\.)*"', "STR_LITERAL", code)
    code = re.sub(r"'(?:[^'\\]|\\.)*'", "STR_LITERAL", code)
    code = re.sub(r"`(?:[^`\\]|\\.)*`", "STR_LITERAL", code)  # template literals
    return code


def normalize_identifiers(code):
    """
    Replace all user-defined identifiers (variable names, function names)
    with generic tokens. This catches plagiarism where someone just
    renames variables.

    Strategy: Replace any identifier that is NOT a language keyword
    with a positional placeholder (ID_0, ID_1, ...) based on first
    occurrence order.
    """
    # Extract all identifiers (words that aren't keywords or numbers)
    identifier_map = {}
    counter = [0]  # Use list for closure mutability

    def replace_identifier(match):
        word = match.group(0)
        # Keep keywords, numbers, and special tokens as-is
        if word.lower() in LANGUAGE_KEYWORDS:
            return word.lower()  # Normalize keyword casing
        if word.startswith("STR_LITERAL") or word.startswith("NUM_"):
            return word
        if re.match(r"^\d+$", word):
            return "NUM_LITERAL"
        # Map this identifier to a consistent placeholder
        if word not in identifier_map:
            identifier_map[word] = f"ID_{counter[0]}"
            counter[0] += 1
        return identifier_map[word]

    return re.sub(r"\b[a-zA-Z_]\w*\b", replace_identifier, code)


def normalize_numbers(code):
    """Replace all numeric literals with a placeholder."""
    code = re.sub(r"\b\d+\.?\d*\b", "NUM_LITERAL", code)
    return code


def extract_logic_tokens(code):
    """
    Extract only MEANINGFUL tokens from code.
    Strips all syntactic noise ({, }, ;, etc.) and focuses on
    the logical structure: identifiers, keywords, operators.
    """
    # First, do full preprocessing
    code = strip_comments(code)
    code = strip_string_literals(code)
    code = normalize_numbers(code)
    code = normalize_identifiers(code)

    # Tokenize: extract words and meaningful operators
    # Keep: identifiers, keywords, operators like ==, !=, <=, >=, &&, ||, +=
    tokens = re.findall(
        r"[a-zA-Z_]\w*|"  # identifiers and keywords
        r"[+\-*/%]=?|"  # arithmetic operators
        r"[<>!=]=|"  # comparison operators
        r"&&|\|\||"  # logical operators
        r"<<|>>|"  # bitwise shift
        r"\+\+|--|"  # increment/decrement
        r"=>|"  # arrow function
        r"[=!<>]",  # single-char operators
        code,
    )

    # Filter out single-character noise that slipped through
    tokens = [t for t in tokens if t not in SYNTAX_NOISE and len(t.strip()) > 0]

    return tokens


def extract_logic_lines(code):
    """
    Extract logical lines of code, cleaned and normalized.
    Used for line-by-line structural comparison.
    """
    code = strip_comments(code)
    code = strip_string_literals(code)

    lines = []
    for line in code.split("\n"):
        # Remove all syntax noise characters
        cleaned = line.strip()
        for ch in SYNTAX_NOISE:
            cleaned = cleaned.replace(ch, " ")
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        # Only keep lines with actual logic (not empty or brace-only)
        if cleaned and len(cleaned) > 2:
            lines.append(cleaned)
    return lines


def _chunk_text(text, chunk_size, overlap):
    """Split long text into overlapping chunks for embedding."""
    text = text.strip()
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]
    chunks = []
    step = max(chunk_size - overlap, 1)
    for i in range(0, len(text), step):
        chunk = text[i : i + chunk_size]
        if chunk.strip():
            chunks.append(chunk)
    return chunks


def _get_embedding_model():
    """Lazily load the transformer model to avoid slow startup."""
    global _EMBED_MODEL, _EMBED_TOKENIZER

    if _EMBED_MODEL is None or _EMBED_TOKENIZER is None:
        torch.set_grad_enabled(False)
        _EMBED_TOKENIZER = AutoTokenizer.from_pretrained(EMBED_MODEL_NAME)
        _EMBED_MODEL = AutoModel.from_pretrained(EMBED_MODEL_NAME)
        _EMBED_MODEL.to(EMBED_DEVICE)
        _EMBED_MODEL.eval()

    return _EMBED_TOKENIZER, _EMBED_MODEL


def _embedding_cache_get(key):
    cached = _EMBED_CACHE.get(key)
    if cached is not None:
        _EMBED_CACHE.move_to_end(key)
    return cached


def _embedding_cache_set(key, value):
    _EMBED_CACHE[key] = value
    _EMBED_CACHE.move_to_end(key)
    if len(_EMBED_CACHE) > EMBED_CACHE_SIZE:
        _EMBED_CACHE.popitem(last=False)


def _prepare_code_for_embedding(code):
    """Reduce comment/string noise while keeping core code structure."""
    code = strip_comments(code)
    code = strip_string_literals(code)
    code = normalize_numbers(code)
    return code


def compute_code_embedding(code):
    """Generate a normalized embedding vector for code."""
    cleaned = _prepare_code_for_embedding(code)
    if not cleaned.strip():
        return None

    cache_key = hashlib.sha256(cleaned.encode("utf-8")).hexdigest()
    cached = _embedding_cache_get(cache_key)
    if cached is not None:
        return cached

    tokenizer, model = _get_embedding_model()
    chunks = _chunk_text(cleaned, EMBED_CHUNK_CHARS, EMBED_CHUNK_OVERLAP)
    if not chunks:
        return None

    embeddings = []
    with torch.no_grad():
        for chunk in chunks:
            inputs = tokenizer(
                chunk,
                return_tensors="pt",
                truncation=True,
                max_length=EMBED_MAX_LENGTH,
            )
            inputs = {k: v.to(EMBED_DEVICE) for k, v in inputs.items()}
            outputs = model(**inputs)
            last_hidden = outputs.last_hidden_state
            mask = inputs["attention_mask"].unsqueeze(-1).expand(last_hidden.size()).float()
            pooled = (last_hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
            embeddings.append(pooled.squeeze(0).cpu())

    if not embeddings:
        return None

    avg_embedding = torch.stack(embeddings).mean(dim=0)
    avg_embedding = avg_embedding / (avg_embedding.norm(p=2) + 1e-12)
    vector = avg_embedding.tolist()
    _embedding_cache_set(cache_key, vector)
    return vector


def compute_embedding_similarity(code1, code2):
    """Cosine similarity between transformer embeddings."""
    emb1 = compute_code_embedding(code1)
    emb2 = compute_code_embedding(code2)
    if not emb1 or not emb2:
        return 0.0

    dot = sum(a * b for a, b in zip(emb1, emb2))
    mag1 = math.sqrt(sum(a * a for a in emb1))
    mag2 = math.sqrt(sum(b * b for b in emb2))
    if mag1 == 0 or mag2 == 0:
        return 0.0
    return round((dot / (mag1 * mag2)) * 100, 2)


# ============================================================
# 2. TF-IDF COSINE SIMILARITY (from scratch, no scikit-learn)
# ============================================================


def compute_tf(tokens):
    """Compute Term Frequency for a list of tokens."""
    tf = Counter(tokens)
    total = len(tokens)
    if total == 0:
        return {}
    return {word: count / total for word, count in tf.items()}


def compute_idf(doc_list):
    """Compute Inverse Document Frequency across multiple documents."""
    idf = {}
    n_docs = len(doc_list)
    all_tokens = set()
    for doc in doc_list:
        all_tokens.update(set(doc))

    for token in all_tokens:
        # Count how many documents contain this token
        containing = sum(1 for doc in doc_list if token in doc)
        idf[token] = math.log((n_docs + 1) / (containing + 1)) + 1  # Smoothed IDF

    return idf


def compute_tfidf_cosine(tokens1, tokens2):
    """
    Compute cosine similarity using TF-IDF on CODE TOKENS (not raw text).

    This is fundamentally different from the old version because:
    - Input is pre-tokenized with noise stripped
    - Variable names are normalized
    - Only meaningful tokens are compared
    """
    if not tokens1 or not tokens2:
        return 0.0

    # Compute TF for each document
    tf1 = compute_tf(tokens1)
    tf2 = compute_tf(tokens2)

    # Compute IDF across both documents
    idf = compute_idf([tokens1, tokens2])

    # Build TF-IDF vectors
    all_tokens = set(list(tf1.keys()) + list(tf2.keys()))

    vec1 = []
    vec2 = []
    for token in all_tokens:
        vec1.append(tf1.get(token, 0) * idf.get(token, 0))
        vec2.append(tf2.get(token, 0) * idf.get(token, 0))

    # Cosine similarity
    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    magnitude1 = math.sqrt(sum(a * a for a in vec1))
    magnitude2 = math.sqrt(sum(b * b for b in vec2))

    if magnitude1 == 0 or magnitude2 == 0:
        return 0.0

    similarity = dot_product / (magnitude1 * magnitude2)
    return round(similarity * 100, 2)


# ============================================================
# 3. N-GRAM FINGERPRINTING (Winnowing Algorithm)
# ============================================================


def get_ngrams(tokens, n=4):
    """Generate n-grams from a token list."""
    if len(tokens) < n:
        return [tuple(tokens)] if tokens else []
    return [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


def hash_ngram(ngram):
    """Create a numeric hash for an n-gram."""
    text = " ".join(str(t) for t in ngram)
    return int(hashlib.md5(text.encode()).hexdigest()[:8], 16)


def winnow_fingerprints(hashes, window_size=4):
    """
    Winnowing algorithm — select minimum hash from each window.
    This creates a compact fingerprint that is robust against
    code insertion/deletion.
    """
    if not hashes:
        return set()
    if len(hashes) <= window_size:
        return {min(hashes)}

    fingerprints = set()
    for i in range(len(hashes) - window_size + 1):
        window = hashes[i : i + window_size]
        fingerprints.add(min(window))
    return fingerprints


def compute_ngram_similarity(tokens1, tokens2, n=4):
    """
    Compare code using n-gram fingerprinting.

    This catches structural plagiarism — when someone copies the
    LOGIC of code (same sequence of operations) even if they rename
    everything. It compares sequences of n consecutive tokens.
    """
    # Generate n-grams
    ngrams1 = get_ngrams(tokens1, n)
    ngrams2 = get_ngrams(tokens2, n)

    if not ngrams1 or not ngrams2:
        return 0.0

    # Hash the n-grams
    hashes1 = [hash_ngram(ng) for ng in ngrams1]
    hashes2 = [hash_ngram(ng) for ng in ngrams2]

    # Get fingerprints using winnowing
    fp1 = winnow_fingerprints(hashes1)
    fp2 = winnow_fingerprints(hashes2)

    if not fp1 or not fp2:
        return 0.0

    # Jaccard similarity on fingerprints
    intersection = fp1 & fp2
    union = fp1 | fp2

    return round((len(intersection) / len(union)) * 100, 2) if union else 0.0


# ============================================================
# 4. AST-AWARE SEQUENCE MATCHING
# ============================================================


def compute_structural_similarity(code1, code2):
    """
    Compare the logical STRUCTURE of code, not the raw text.

    Instead of comparing characters (which flags '{' and '}' as matches),
    this compares cleaned logical lines. Only meaningful code lines
    are compared.
    """
    lines1 = extract_logic_lines(code1)
    lines2 = extract_logic_lines(code2)

    if not lines1 or not lines2:
        return 0.0, []

    # Use SequenceMatcher on LOGICAL LINES, not raw characters
    matcher = difflib.SequenceMatcher(None, lines1, lines2)
    ratio = matcher.ratio()

    # Extract meaningful matching blocks
    matching_blocks = []
    for block in matcher.get_matching_blocks():
        if block.size >= 2:  # At least 2 consecutive matching lines
            matched_lines = lines1[block.a : block.a + block.size]
            matched_text = "\n".join(matched_lines)

            # Skip trivially short matches
            if len(matched_text.strip()) < 15:
                continue

            matching_blocks.append(
                {
                    "code1_start": block.a + 1,
                    "code1_end": block.a + block.size,
                    "code2_start": block.b + 1,
                    "code2_end": block.b + block.size,
                    "lines_matched": block.size,
                    "text": matched_text[:500],
                }
            )

    return round(ratio * 100, 2), matching_blocks


# ============================================================
# 5. IDENTIFIER PATTERN ANALYSIS
# ============================================================


def extract_code_patterns(code):
    """
    Extract high-level code patterns for comparison:
    - Function signatures
    - Loop structures
    - Conditional chains
    - Data structure usage
    """
    code = strip_comments(code)
    patterns = []

    # Function definitions
    func_defs = re.findall(
        r"(?:function\s+\w+|def\s+\w+|(?:public|private|protected)?\s*(?:static\s+)?\w+\s+\w+)\s*\([^)]*\)",
        code,
    )
    for f in func_defs:
        # Normalize: replace specific names with placeholders
        normalized = re.sub(
            r"\b[a-zA-Z_]\w*\b",
            lambda m: m.group() if m.group().lower() in LANGUAGE_KEYWORDS else "IDENT",
            f,
        )
        patterns.append(("FUNC", normalized))

    # Loop patterns
    loops = re.findall(r"(for|while|do)\s*[\(\{]", code)
    for l in loops:
        patterns.append(("LOOP", l))

    # Conditional patterns
    conds = re.findall(r"(if|elif|else if|else|switch)\s*[\(\{]?", code)
    for c in conds:
        patterns.append(("COND", c))

    # Return patterns
    returns = re.findall(r"return\s+.{1,50}", code)
    for r in returns:
        normalized = re.sub(
            r"\b[a-zA-Z_]\w*\b",
            lambda m: m.group() if m.group().lower() in LANGUAGE_KEYWORDS else "IDENT",
            r,
        )
        patterns.append(("RETURN", normalized))

    return patterns


def compute_pattern_similarity(code1, code2):
    """
    Compare the high-level structural patterns of two code files.
    This catches cases where someone copies the algorithm structure
    but rewrites individual lines.
    """
    patterns1 = extract_code_patterns(code1)
    patterns2 = extract_code_patterns(code2)

    if not patterns1 or not patterns2:
        return 0.0

    # Convert to string representations for comparison
    str_patterns1 = [f"{p[0]}:{p[1]}" for p in patterns1]
    str_patterns2 = [f"{p[0]}:{p[1]}" for p in patterns2]

    # Sequence matching on pattern lists
    matcher = difflib.SequenceMatcher(None, str_patterns1, str_patterns2)
    return round(matcher.ratio() * 100, 2)


def _scores_to_feature_vector(scores):
    """Convert similarity scores to a normalized feature vector."""
    features = []
    for name in PLAGIARISM_MODEL_META["features"]:
        value = scores.get(name, 0) / 100.0
        features.append(value)
    return torch.tensor(features, dtype=torch.float32)


def load_plagiarism_model():
    """Load the optional calibrator model from disk."""
    global PLAGIARISM_MODEL
    if PLAGIARISM_MODEL is not None:
        return PLAGIARISM_MODEL
    if not os.path.exists(PLAGIARISM_MODEL_PATH):
        return None

    with open(PLAGIARISM_MODEL_PATH, "r", encoding="utf-8") as handle:
        data = json.load(handle)

    if data.get("version") != PLAGIARISM_MODEL_META["version"]:
        return None
    if data.get("features") != PLAGIARISM_MODEL_META["features"]:
        return None

    PLAGIARISM_MODEL = data
    return PLAGIARISM_MODEL


def save_plagiarism_model(model_data):
    """Persist the calibrator model to disk."""
    with open(PLAGIARISM_MODEL_PATH, "w", encoding="utf-8") as handle:
        json.dump(model_data, handle, indent=2)


def predict_plagiarism_probability(scores):
    """Predict plagiarism probability using the calibrator model."""
    model = load_plagiarism_model()
    if not model:
        return None

    weights = model.get("weights", {})
    bias = float(model.get("bias", 0.0))

    logit = bias
    for name in PLAGIARISM_MODEL_META["features"]:
        weight = float(weights.get(name, 0.0))
        logit += (scores.get(name, 0.0) / 100.0) * weight

    probability = 1 / (1 + math.exp(-logit))
    return round(probability * 100, 2)


def train_plagiarism_calibrator(pairs, epochs=200, lr=0.1, seed=42):
    """Train a logistic calibrator using labeled code pairs."""
    torch.manual_seed(seed)

    feature_rows = []
    labels = []
    for pair in pairs:
        code1 = pair.get("code1", "")
        code2 = pair.get("code2", "")
        label = pair.get("label")
        if label not in (0, 1):
            continue

        scores, _ = compute_similarity_scores(code1, code2)
        feature_rows.append(_scores_to_feature_vector(scores))
        labels.append([float(label)])

    if len(feature_rows) < 4:
        return None, "Need at least 4 valid labeled pairs to train."

    X = torch.stack(feature_rows)
    y = torch.tensor(labels, dtype=torch.float32)

    weights = torch.zeros((X.shape[1], 1), dtype=torch.float32, requires_grad=True)
    bias = torch.zeros((1,), dtype=torch.float32, requires_grad=True)
    optimizer = torch.optim.Adam([weights, bias], lr=lr)

    for _ in range(epochs):
        logits = X @ weights + bias
        loss = F.binary_cross_entropy_with_logits(logits, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        logits = X @ weights + bias
        probs = torch.sigmoid(logits)
        preds = (probs >= 0.5).float()
        accuracy = (preds == y).float().mean().item() * 100
        final_loss = F.binary_cross_entropy_with_logits(logits, y).item()

    model_data = {
        "version": PLAGIARISM_MODEL_META["version"],
        "features": PLAGIARISM_MODEL_META["features"],
        "weights": {
            name: float(weights[i].item())
            for i, name in enumerate(PLAGIARISM_MODEL_META["features"])
        },
        "bias": float(bias.item()),
        "metrics": {
            "training_loss": round(final_loss, 6),
            "training_accuracy": round(accuracy, 2),
            "samples": len(feature_rows),
        },
    }

    return model_data, None


def compute_similarity_scores(raw_code1, raw_code2):
    """Compute similarity scores used by the heuristics and ML calibrator."""
    tokens1 = extract_logic_tokens(raw_code1)
    tokens2 = extract_logic_tokens(raw_code2)

    cosine_score = compute_tfidf_cosine(tokens1, tokens2)
    ngram_score = compute_ngram_similarity(tokens1, tokens2, n=4)
    structural_score, matching_blocks = compute_structural_similarity(
        raw_code1, raw_code2
    )
    pattern_score = compute_pattern_similarity(raw_code1, raw_code2)
    embedding_score = compute_embedding_similarity(raw_code1, raw_code2)

    scores = {
        "cosine_similarity": cosine_score,
        "ngram_fingerprint": ngram_score,
        "structural_similarity": structural_score,
        "pattern_similarity": pattern_score,
        "embedding_similarity": embedding_score,
    }

    return scores, {
        "tokens1": tokens1,
        "tokens2": tokens2,
        "matching_blocks": matching_blocks,
    }


# ============================================================
# MAIN ANALYSIS PIPELINE
# ============================================================


def analyze_code_pair(raw_code1, raw_code2):
    """
    Run the full analysis pipeline on two pieces of code.

    Returns a comprehensive similarity report with 5 different
    algorithm scores and an intelligent overall verdict.
    """
    scores, context = compute_similarity_scores(raw_code1, raw_code2)

    # ---- Intelligent weighted scoring ----
    # Cosine: overall vocabulary overlap                    -> 20%
    # N-gram: copied logic sequences                        -> 30%
    # Structural: line-level matching                       -> 20%
    # Pattern: algorithm-level structure                    -> 10%
    # Embedding: semantic similarity                         -> 20%
    overall = round(
        scores["cosine_similarity"] * 0.20
        + scores["ngram_fingerprint"] * 0.30
        + scores["structural_similarity"] * 0.20
        + scores["pattern_similarity"] * 0.10
        + scores["embedding_similarity"] * 0.20,
        2,
    )

    # Determine verdict
    if overall >= 60:
        verdict = "High"
    elif overall >= 30:
        verdict = "Medium"
    else:
        verdict = "Low"

    # Token stats for transparency
    tokens1 = context["tokens1"]
    tokens2 = context["tokens2"]
    matching_blocks = context["matching_blocks"]

    stats = {
        "tokens_file1": len(tokens1),
        "tokens_file2": len(tokens2),
        "unique_tokens_file1": len(set(tokens1)),
        "unique_tokens_file2": len(set(tokens2)),
        "sample_tokens_file1": tokens1[:20],
        "sample_tokens_file2": tokens2[:20],
    }

    ml_probability = predict_plagiarism_probability(scores)
    if ml_probability is None:
        ml_verdict = "Not Trained"
    elif ml_probability >= 70:
        ml_verdict = "High"
    elif ml_probability >= 40:
        ml_verdict = "Medium"
    else:
        ml_verdict = "Low"

    return {
        "cosine_similarity": scores["cosine_similarity"],
        "ngram_fingerprint": scores["ngram_fingerprint"],
        "structural_similarity": scores["structural_similarity"],
        "pattern_similarity": scores["pattern_similarity"],
        "embedding_similarity": scores["embedding_similarity"],
        "overall": overall,
        "verdict": verdict,
        "ml_probability": ml_probability,
        "ml_verdict": ml_verdict,
        "matching_blocks": matching_blocks,
        "analysis_stats": stats,
    }


# ============================================================
# API ENDPOINTS
# ============================================================


@app.route("/analyze", methods=["POST"])
def analyze():
    """
    POST /analyze

    Accepts two code strings and returns deep similarity analysis.

    Request body (JSON):
    {
        "code1": "function hello() { ... }",
        "code2": "function greet() { ... }"
    }
    """
    data = request.get_json()

    if not data or "code1" not in data or "code2" not in data:
        return jsonify(
            {"error": "Please provide 'code1' and 'code2' in the request body."}
        ), 400

    code1 = data["code1"]
    code2 = data["code2"]

    if not code1.strip() or not code2.strip():
        return jsonify(
            {
                "cosine_similarity": 0,
                "ngram_fingerprint": 0,
                "structural_similarity": 0,
                "pattern_similarity": 0,
                "embedding_similarity": 0,
                "overall": 0,
                "verdict": "Low",
                "ml_probability": None,
                "ml_verdict": "Not Trained",
                "matching_blocks": [],
                "analysis_stats": {},
            }
        )

    result = analyze_code_pair(code1, code2)
    return jsonify(result)


@app.route("/train-calibrator", methods=["POST"])
def train_calibrator():
    """
    POST /train-calibrator

    Train the optional ML calibrator with labeled code pairs.

    Request body (JSON):
    {
        "pairs": [
            {"code1": "...", "code2": "...", "label": 1},
            {"code1": "...", "code2": "...", "label": 0}
        ],
        "epochs": 200,
        "lr": 0.1,
        "seed": 42,
        "save": true
    }
    """
    global PLAGIARISM_MODEL

    data = request.get_json()
    if not data or "pairs" not in data:
        return jsonify({"error": "Please provide 'pairs' in the request body."}), 400

    pairs = data.get("pairs")
    if not isinstance(pairs, list) or len(pairs) < 4:
        return jsonify({"error": "Please provide at least 4 labeled pairs."}), 400

    epochs = int(data.get("epochs", 200))
    lr = float(data.get("lr", 0.1))
    seed = int(data.get("seed", 42))
    save_model = bool(data.get("save", True))

    model_data, error = train_plagiarism_calibrator(
        pairs, epochs=epochs, lr=lr, seed=seed
    )
    if error:
        return jsonify({"error": error}), 400

    if save_model:
        save_plagiarism_model(model_data)
        PLAGIARISM_MODEL = model_data

    return jsonify(
        {
            "trained": True,
            "saved": save_model,
            "model_path": PLAGIARISM_MODEL_PATH if save_model else None,
            "features": model_data.get("features"),
            "metrics": model_data.get("metrics"),
        }
    )


@app.route("/model-info", methods=["GET"])
def model_info():
    """Return embedding model and calibrator status."""
    model = load_plagiarism_model()
    return jsonify(
        {
            "embedding_model": EMBED_MODEL_NAME,
            "embedding_device": EMBED_DEVICE,
            "embedding_cache_size": EMBED_CACHE_SIZE,
            "calibrator_loaded": model is not None,
            "calibrator_path": PLAGIARISM_MODEL_PATH,
            "calibrator_features": PLAGIARISM_MODEL_META["features"],
            "calibrator_metrics": model.get("metrics") if model else None,
        }
    )


# ============================================================
# 6. AI CODE DETECTION ENGINE
# ============================================================
# Based on research into AI-generated code fingerprints:
# - Excessive/obvious commenting
# - Overly verbose & consistent naming
# - Perfect uniform formatting
# - Boilerplate-heavy patterns (excessive try/catch)
# - Low "burstiness" (uniform line complexity)
# - Generic variable names (result, data, output, temp)
# - Formulaic function structure
# - Over-documentation with docstrings
# - Predictable error handling
# ============================================================

# Phrases AI loves to put in comments that humans rarely write
AI_COMMENT_PHRASES = [
    "this function",
    "this method",
    "this class",
    "this variable",
    "here we",
    "we use",
    "we need",
    "we can",
    "we will",
    "the following",
    "as follows",
    "note that",
    "note:",
    "initialize",
    "initialise",
    "define the",
    "create a",
    "check if",
    "check whether",
    "ensure that",
    "make sure",
    "return the",
    "returns the",
    "returns a",
    "return a",
    "set the",
    "get the",
    "set up",
    "sets up",
    "handle the",
    "handles the",
    "process the",
    "processes the",
    "loop through",
    "iterate over",
    "iterate through",
    "for each",
    "for every",
    "example usage",
    "example:",
    "e.g.",
    "step 1",
    "step 2",
    "step 3",
    "first,",
    "second,",
    "third,",
    "finally,",
    "parameters:",
    "args:",
    "arguments:",
    "todo:",
    "fixme:",
    "hack:",
]

# Generic variable names AI tends to use excessively
AI_GENERIC_NAMES = [
    "result",
    "results",
    "data",
    "output",
    "input",
    "temp",
    "tmp",
    "value",
    "values",
    "item",
    "items",
    "element",
    "elements",
    "response",
    "request",
    "callback",
    "handler",
    "listener",
    "config",
    "options",
    "params",
    "args",
    "kwargs",
    "obj",
    "arr",
    "lst",
    "dict",
    "map",
    "err",
    "error",
    "exception",
    "ex",
    "count",
    "index",
    "idx",
    "key",
    "val",
    "flag",
    "status",
    "state",
    "current",
    "prev",
    "next",
    "first",
    "last",
]

# Boilerplate patterns AI over-uses
AI_BOILERPLATE_PATTERNS = [
    r"try\s*[{:]",  # Excessive try blocks
    r"catch\s*\(",  # Excessive catch blocks
    r"except\s+\w*Exception",  # Python exception catching
    r"console\.log\(",  # Debug logging left in
    r"print\(",  # Python print debugging
    r"TODO",  # TODO comments
    r"FIXME",  # FIXME comments
    r"if\s*\(\s*!\s*\w+\s*\)",  # Null/undefined checks
    r"if\s+not\s+\w+\s*:",  # Python none checks
    r"===\s*null|===\s*undefined",  # Strict null checks
    r"is\s+None",  # Python None checks
    r"throw\s+new\s+Error",  # Generic error throwing
    r"raise\s+\w*Error",  # Python error raising
]


def detect_ai_patterns(code):
    """
    Analyze a single code file for AI-generated code patterns.

    Returns a detailed report with individual scores for each
    heuristic and an overall AI probability percentage.
    """
    if not code or not code.strip():
        return {
            "ai_probability": 0,
            "verdict": "Unable to analyze",
            "indicators": [],
            "details": {},
        }

    lines = code.split("\n")
    total_lines = len(lines)
    non_empty_lines = [l for l in lines if l.strip()]
    total_non_empty = len(non_empty_lines) or 1  # Avoid division by zero

    indicators = []
    scores = {}

    # ---- HEURISTIC 1: Comment Density & Obviousness ----
    comment_lines = []
    for line in lines:
        stripped = line.strip()
        if (
            stripped.startswith("//")
            or stripped.startswith("#")
            or stripped.startswith("*")
        ):
            comment_lines.append(stripped.lower())
        elif (
            stripped.startswith("/*")
            or stripped.startswith('"""')
            or stripped.startswith("'''")
        ):
            comment_lines.append(stripped.lower())

    comment_ratio = len(comment_lines) / total_non_empty
    # AI typically comments 30-50% of lines; humans comment 5-15%
    comment_score = min(100, max(0, (comment_ratio - 0.10) * 250))

    # Check for AI-style comment phrases
    ai_phrase_count = 0
    for comment in comment_lines:
        for phrase in AI_COMMENT_PHRASES:
            if phrase in comment:
                ai_phrase_count += 1
                break

    ai_phrase_ratio = ai_phrase_count / max(len(comment_lines), 1)
    comment_quality_score = min(100, ai_phrase_ratio * 150)

    combined_comment_score = comment_score * 0.4 + comment_quality_score * 0.6
    scores["commenting"] = round(combined_comment_score, 1)

    if combined_comment_score > 50:
        indicators.append(
            {
                "pattern": "Excessive/Obvious Commenting",
                "severity": "high" if combined_comment_score > 70 else "medium",
                "description": f"Found {len(comment_lines)} comment lines ({comment_ratio:.0%} of code). "
                f"{ai_phrase_count} use AI-typical phrasing like 'This function...', 'Here we...'",
                "score": scores["commenting"],
            }
        )

    # ---- HEURISTIC 2: Naming Convention Consistency ----
    identifiers = re.findall(r"\b[a-zA-Z_]\w{2,}\b", code)
    # Filter out keywords
    identifiers = [
        w for w in identifiers if w.lower() not in LANGUAGE_KEYWORDS and not w.isupper()
    ]  # Skip constants

    if identifiers:
        # Check naming style consistency
        camel_count = sum(1 for w in identifiers if re.match(r"^[a-z]+[A-Z]", w))
        snake_count = sum(1 for w in identifiers if "_" in w and w.islower())
        pascal_count = sum(1 for w in identifiers if re.match(r"^[A-Z][a-z]+[A-Z]", w))

        total_styled = camel_count + snake_count + pascal_count
        if total_styled > 0:
            # AI uses ONE style 95%+ of the time; humans mix styles
            dominant_ratio = max(camel_count, snake_count, pascal_count) / total_styled
            naming_consistency_score = min(100, max(0, (dominant_ratio - 0.7) * 333))
        else:
            naming_consistency_score = 30  # Neutral

        # Check for verbose names (AI loves long descriptive names)
        avg_name_length = sum(len(w) for w in identifiers) / len(identifiers)
        verbose_score = min(100, max(0, (avg_name_length - 8) * 20))

        # Check for generic AI-favorite names
        generic_count = sum(1 for w in identifiers if w.lower() in AI_GENERIC_NAMES)
        generic_ratio = generic_count / len(identifiers)
        generic_score = min(100, generic_ratio * 300)

        naming_score = (
            naming_consistency_score * 0.3 + verbose_score * 0.3 + generic_score * 0.4
        )
        scores["naming"] = round(naming_score, 1)

        if naming_score > 40:
            indicators.append(
                {
                    "pattern": "AI-Typical Naming Patterns",
                    "severity": "high" if naming_score > 65 else "medium",
                    "description": f"Avg name length: {avg_name_length:.1f} chars. "
                    f"Naming consistency: {dominant_ratio:.0%}. "
                    f"{generic_count}/{len(identifiers)} identifiers are generic AI-favorites "
                    f"(result, data, output, temp, etc.)",
                    "score": scores["naming"],
                }
            )
    else:
        scores["naming"] = 0

    # ---- HEURISTIC 3: Formatting Uniformity (Low Burstiness) ----
    if len(non_empty_lines) > 3:
        line_lengths = [len(l) for l in non_empty_lines]
        avg_length = sum(line_lengths) / len(line_lengths)
        variance = sum((l - avg_length) ** 2 for l in line_lengths) / len(line_lengths)
        std_dev = math.sqrt(variance)

        # Low std_dev = very uniform = likely AI
        # Human code typically has std_dev > 20; AI often < 15
        coefficient_of_variation = std_dev / max(avg_length, 1)
        uniformity_score = min(100, max(0, (1 - coefficient_of_variation) * 120 - 20))

        # Check indentation consistency
        indentation_levels = []
        for line in non_empty_lines:
            leading = len(line) - len(line.lstrip())
            indentation_levels.append(leading)

        unique_indents = len(set(indentation_levels))
        indent_ratio = unique_indents / len(non_empty_lines)
        # AI uses very few indent levels consistently
        indent_score = min(100, max(0, (1 - indent_ratio) * 130 - 10))

        formatting_score = uniformity_score * 0.6 + indent_score * 0.4
        scores["formatting"] = round(formatting_score, 1)

        if formatting_score > 50:
            indicators.append(
                {
                    "pattern": "Unnaturally Uniform Formatting",
                    "severity": "medium",
                    "description": f"Line length std deviation: {std_dev:.1f} (AI typical: <15, Human typical: >20). "
                    f"Code appears unusually consistent in structure.",
                    "score": scores["formatting"],
                }
            )
    else:
        scores["formatting"] = 0

    # ---- HEURISTIC 4: Boilerplate Pattern Density ----
    boilerplate_hits = 0
    for pattern in AI_BOILERPLATE_PATTERNS:
        matches = re.findall(pattern, code)
        boilerplate_hits += len(matches)

    boilerplate_ratio = boilerplate_hits / total_non_empty
    boilerplate_score = min(100, boilerplate_ratio * 500)
    scores["boilerplate"] = round(boilerplate_score, 1)

    if boilerplate_score > 30:
        indicators.append(
            {
                "pattern": "Excessive Boilerplate Code",
                "severity": "medium" if boilerplate_score < 60 else "high",
                "description": f"Found {boilerplate_hits} boilerplate patterns (try/catch, null checks, "
                f"generic error handling) in {total_non_empty} lines.",
                "score": scores["boilerplate"],
            }
        )

    # ---- HEURISTIC 5: Docstring/JSDoc Density ----
    docstring_patterns = [
        r"/\*\*[\s\S]*?\*/",  # JSDoc /** ... */
        r'"""[\s\S]*?"""',  # Python docstrings
        r"'''[\s\S]*?'''",  # Python docstrings
        r"^\s*\*\s+@param",  # @param tags
        r"^\s*\*\s+@returns?",  # @return tags
        r"^\s*:param\s+",  # Python :param
        r"^\s*:returns?:",  # Python :returns
        r"^\s*Args:",  # Google-style Args:
        r"^\s*Returns:",  # Google-style Returns:
    ]

    docstring_count = 0
    for pattern in docstring_patterns:
        docstring_count += len(re.findall(pattern, code, re.MULTILINE))

    # Count function definitions to get ratio
    func_count = len(
        re.findall(
            r"(?:function\s+\w+|def\s+\w+|(?:const|let|var)\s+\w+\s*=\s*(?:async\s+)?(?:\([^)]*\)|[a-zA-Z_]\w*)\s*=>)",
            code,
        )
    )
    func_count = max(func_count, 1)

    doc_per_func = docstring_count / func_count
    # AI documents EVERY function; humans document selectively
    docstring_score = min(100, doc_per_func * 40)
    scores["documentation"] = round(docstring_score, 1)

    if docstring_score > 40:
        indicators.append(
            {
                "pattern": "Over-Documentation",
                "severity": "medium",
                "description": f"Found {docstring_count} documentation blocks for {func_count} functions. "
                f"AI tends to document every function with full parameter descriptions.",
                "score": scores["documentation"],
            }
        )

    # ---- HEURISTIC 6: Sequential Step Pattern ----
    step_patterns = re.findall(
        r"(?://|#)\s*(?:Step|Phase|Stage)\s*\d+|"
        r"(?://|#)\s*(?:\d+[\.\)]\s+\w+)|"
        r"(?://|#)\s*(?:First|Second|Third|Next|Then|Finally)",
        code,
        re.IGNORECASE,
    )
    step_score = min(100, len(step_patterns) * 25)
    scores["sequential_steps"] = round(step_score, 1)

    if step_score > 0:
        indicators.append(
            {
                "pattern": "Sequential Step Comments",
                "severity": "low" if step_score < 50 else "medium",
                "description": f"Found {len(step_patterns)} sequential step markers "
                f"('Step 1', 'First...', 'Next...'). AI frequently structures code in numbered steps.",
                "score": scores["sequential_steps"],
            }
        )

    # ---- HEURISTIC 7: Repetitive Structure ----
    # Check if function bodies follow the same template
    func_bodies = re.findall(
        r"(?:function\s+\w+\s*\([^)]*\)\s*\{|def\s+\w+\s*\([^)]*\)\s*:)([\s\S]*?)(?=function\s+\w+|def\s+\w+|$)",
        code,
    )

    if len(func_bodies) >= 3:
        # Compare structure of function bodies using their token patterns
        body_patterns = []
        for body in func_bodies:
            # Extract structural tokens only (keywords + operators)
            tokens = re.findall(
                r"\b(?:if|else|for|while|return|try|catch|except|const|let|var)\b|[=<>!+\-*/]",
                body,
            )
            body_patterns.append(" ".join(tokens))

        # Check similarity between function structures
        if body_patterns:
            similarities = []
            for i in range(len(body_patterns)):
                for j in range(i + 1, len(body_patterns)):
                    if body_patterns[i] and body_patterns[j]:
                        matcher = difflib.SequenceMatcher(
                            None, body_patterns[i], body_patterns[j]
                        )
                        similarities.append(matcher.ratio())

            if similarities:
                avg_similarity = sum(similarities) / len(similarities)
                repetitive_score = min(100, max(0, (avg_similarity - 0.3) * 143))
                scores["repetitive_structure"] = round(repetitive_score, 1)

                if repetitive_score > 40:
                    indicators.append(
                        {
                            "pattern": "Repetitive Function Structure",
                            "severity": "medium",
                            "description": f"Function bodies share {avg_similarity:.0%} structural similarity. "
                            f"AI tends to use the same template for every function.",
                            "score": scores["repetitive_structure"],
                        }
                    )
            else:
                scores["repetitive_structure"] = 0
        else:
            scores["repetitive_structure"] = 0
    else:
        scores["repetitive_structure"] = 0

    # ---- HEURISTIC 8: Error Message Quality ----
    error_messages = re.findall(r'(?:Error|Exception)\s*\(\s*["\']([^"\']+)["\']', code)
    generic_error_count = 0
    for msg in error_messages:
        msg_lower = msg.lower()
        if any(
            phrase in msg_lower
            for phrase in [
                "something went wrong",
                "an error occurred",
                "failed to",
                "unable to",
                "invalid",
                "not found",
                "please provide",
                "is required",
                "must be",
                "cannot be",
            ]
        ):
            generic_error_count += 1

    if error_messages:
        error_generic_ratio = generic_error_count / len(error_messages)
        error_score = min(100, error_generic_ratio * 120)
    else:
        error_score = 0
    scores["error_handling"] = round(error_score, 1)

    if error_score > 40:
        indicators.append(
            {
                "pattern": "Generic Error Messages",
                "severity": "low",
                "description": f"{generic_error_count}/{len(error_messages)} error messages use generic AI-style "
                f"phrasing ('Something went wrong', 'Failed to...', 'Invalid...').",
                "score": scores["error_handling"],
            }
        )

    # ---- HEURISTIC 9: Code Cleanliness (Absence of Human Traces) ----
    human_traces = 0
    # Commented-out code (humans leave dead code; AI doesn't)
    commented_code = re.findall(r"(?://|#)\s*\w+\s*[=({\[]", code)
    human_traces += len(commented_code)
    # TODO/FIXME/HACK with personal notes
    personal_notes = re.findall(
        r"(?://|#)\s*(?:TODO|FIXME|HACK|XXX|TEMP)\s*[-:]\s*\w{3,}", code, re.IGNORECASE
    )
    human_traces += len(personal_notes) * 3  # Weight personal notes higher
    # Inconsistent spacing or style breaks
    style_breaks = 0
    prev_indent_char = None
    for line in lines:
        if line and line[0] in (" ", "\t"):
            curr_char = line[0]
            if prev_indent_char and curr_char != prev_indent_char:
                style_breaks += 1
            prev_indent_char = curr_char

    human_traces += style_breaks

    # MORE human traces = LESS likely AI
    human_trace_ratio = human_traces / total_non_empty
    cleanliness_score = min(100, max(0, (1 - human_trace_ratio * 10) * 100))
    scores["cleanliness"] = round(cleanliness_score, 1)

    if cleanliness_score > 60:
        indicators.append(
            {
                "pattern": "Suspiciously Clean Code",
                "severity": "medium" if cleanliness_score < 80 else "high",
                "description": f"Code has very few 'human traces' (commented-out code, personal TODOs, "
                f"style inconsistencies). Found only {human_traces} human markers in {total_non_empty} lines.",
                "score": scores["cleanliness"],
            }
        )

    # ---- HEURISTIC 10: Import/Dependency Pattern ----
    import_lines = re.findall(
        r'(?:import\s+.+|from\s+\w+\s+import|require\s*\(|#include\s*[<"])', code
    )
    if total_non_empty > 10:
        import_ratio = len(import_lines) / total_non_empty
        # AI tends to import exactly what's needed, cleanly organized
        # Check if imports are at the very top (AI always does this)
        first_import_line = None
        last_import_line = None
        for i, line in enumerate(lines):
            stripped = line.strip()
            if re.match(
                r"^(?:import|from|const\s+\w+\s*=\s*require|#include)", stripped
            ):
                if first_import_line is None:
                    first_import_line = i
                last_import_line = i

        if first_import_line is not None and last_import_line is not None:
            import_block_size = last_import_line - first_import_line + 1
            # AI puts all imports in a tight block at the top
            imports_grouped = import_block_size <= len(import_lines) + 2
            import_at_top = first_import_line <= 5
            import_score = 50 if (imports_grouped and import_at_top) else 10
        else:
            import_score = 20
    else:
        import_score = 0
    scores["imports"] = round(import_score, 1)

    # ---- HEURISTIC 11: Type Hint / Annotation Density ----
    type_hint_patterns = [
        r":\s*[A-Z][a-zA-Z0-9_]*\b",  # Python type hints (e.g. : Any, : List)
        r"->\s*[A-Z][a-zA-Z0-9_]*\b",  # Python return hints (e.g. -> None)
        r":\s*(?:string|number|boolean|any|void|unknown)\b",  # TS basic types
        r"<\w+>",  # Generics <T>
        r"as\s+[A-Z]\w+",  # TS type assertions
    ]
    type_hints = 0
    for pattern in type_hint_patterns:
        type_hints += len(re.findall(pattern, code))

    # AI tends to add type hints aggressively. If density per function is very high:
    type_hint_score = 0
    if func_count > 0:
        hints_per_func = type_hints / func_count
        type_hint_score = min(100, hints_per_func * 25)
    scores["type_hints"] = round(type_hint_score, 1)

    if type_hint_score > 40:
        indicators.append(
            {
                "pattern": "High Type Hint Density",
                "severity": "medium",
                "description": f"Found {type_hints} type annotations for {func_count} functions. "
                f"AI often aggressively over-types code even when not required.",
                "score": scores["type_hints"],
            }
        )

    # ---- HEURISTIC 12: Immutability Bias ----
    # Specifically looking at JS/TS, where AI uses 'const' almost exclusively
    const_count = len(re.findall(r"\bconst\s+\w+", code))
    let_var_count = len(re.findall(r"\b(?:let|var)\s+\w+", code))

    immutability_score = 0
    const_ratio = 0
    if const_count + let_var_count > 5:
        const_ratio = const_count / (const_count + let_var_count)
        if const_ratio > 0.90:
            immutability_score = min(
                100, (const_ratio - 0.90) * 1000
            )  # 0.90->0, 1.0->100
    scores["immutability_bias"] = round(immutability_score, 1)

    if immutability_score > 50:
        indicators.append(
            {
                "pattern": "Immutability Bias (const)",
                "severity": "low",
                "description": f"Code uses `const` for {const_ratio:.0%} of variable declarations. "
                f"AI strongly prefers functional/immutable paradigms.",
                "score": scores["immutability_bias"],
            }
        )

    # ---- HEURISTIC 13: Return Statement Verbosity ----
    # Assigning to a variable immediately before returning it
    return_verbosity = len(
        re.findall(
            r"(?:const|let|var)?\s*(\w+)\s*=[^;\n]+[;\n]+\s*return\s+\1\s*[;\n]", code
        )
    )
    return_verb_score = min(100, return_verbosity * 33)
    scores["return_verbosity"] = round(return_verb_score, 1)

    if return_verb_score > 0:
        indicators.append(
            {
                "pattern": "Verbose Return Statements",
                "severity": "low",
                "description": f"Found {return_verbosity} instances of variable assignment immediately followed "
                f"by returning that variable. Typical AI pattern.",
                "score": scores["return_verbosity"],
            }
        )

    # ---- HEURISTIC 14: Over-Modularization ----
    # AI breaks simple code into many tiny functions.
    modularization_score = 0
    avg_func_length = 0
    if func_count > 3:
        avg_func_length = total_non_empty / func_count
        if avg_func_length < 8:  # Very short functions on average
            modularization_score = min(100, (8 - avg_func_length) * 15)
    scores["over_modularization"] = round(modularization_score, 1)

    if modularization_score > 30:
        indicators.append(
            {
                "pattern": "Over-Modularization",
                "severity": "medium",
                "description": f"Code contains {func_count} functions averaging only {avg_func_length:.1f} lines each. "
                f"AI often artificially separates logic into tiny helpers.",
                "score": scores["over_modularization"],
            }
        )

    # ---- HEURISTIC 15: Outdated Idioms / Hallucinations ----
    outdated_patterns = [
        r"from\s+typing\s+import\s+.*(?:List|Dict|Tuple|Set|Optional)",  # Python <3.9 typing
        r"Object\.assign\(\{\}",  # Old JS spread alternative
        r"\.bind\(this\)",  # Pre-arrow function binding
        r'require\s*\(\s*[\'"]react[\'"]\s*\)',  # CJS React imports in modern code
    ]
    outdated_hits = sum(1 for p in outdated_patterns if re.search(p, code))
    outdated_score = min(100, outdated_hits * 50)
    scores["outdated_idioms"] = round(outdated_score, 1)

    if outdated_score > 0:
        indicators.append(
            {
                "pattern": "Outdated Idioms / Hallucinations",
                "severity": "high",
                "description": f"Found {outdated_hits} outdated language idioms (e.g., old typing imports, `.bind(this)`). "
                f"AI models often regurgitate older training data.",
                "score": scores["outdated_idioms"],
            }
        )

    # ---- HEURISTIC 16: Variable Name Entropy ----
    # AI tends to use highly uniform, descriptive names (medium-entropy).
    # Humans use a mix of short aliases + long names (high variance).
    raw_identifiers = re.findall(r"\b[a-zA-Z_][a-zA-Z0-9_]{2,}\b", strip_comments(code))
    raw_identifiers = [w for w in raw_identifiers if w.lower() not in LANGUAGE_KEYWORDS]
    entropy_score = 0
    identifier_length_variance = 0
    if len(raw_identifiers) >= 10:
        import math as _math

        lengths = [len(w) for w in raw_identifiers]
        mean_len = sum(lengths) / len(lengths)
        variance = sum((l - mean_len) ** 2 for l in lengths) / len(lengths)
        identifier_length_variance = round(variance, 2)
        # AI: moderate, very consistent length (low variance, e.g. 5-9)
        # Human: high variance (mix of 'i', 'x' and 'calculateTotalRevenue')
        if variance < 4.0 and mean_len > 5:
            entropy_score = min(100, (4.0 - variance) * 30)
    scores["name_entropy"] = round(entropy_score, 1)

    if entropy_score > 40:
        indicators.append(
            {
                "pattern": "Uniform Variable Name Length",
                "severity": "medium",
                "description": f"Identifier lengths are suspiciously uniform (variance={identifier_length_variance}). "
                f"AI tends to produce consistent, descriptive names; humans mix short aliases with long names.",
                "score": scores["name_entropy"],
            }
        )

    # ---- HEURISTIC 17: Comment-to-Code Ratio ----
    # Humans write a moderate amount of inline comments.
    # AI writes either too many (over-documentation) or too few.
    comment_ratio = len(comment_lines) / total_non_empty if total_non_empty > 0 else 0
    # Over-documentation (>40% comment rate) is a strong AI signal
    if comment_ratio > 0.40:
        comment_ratio_score = min(100, (comment_ratio - 0.40) * 300)
    # Almost no comments in a long file is also suspicious
    elif comment_ratio < 0.03 and total_non_empty > 30:
        comment_ratio_score = min(60, (0.03 - comment_ratio) * 2000)
    else:
        comment_ratio_score = 0
    scores["comment_ratio"] = round(comment_ratio_score, 1)

    if comment_ratio_score > 30:
        direction = "over-documented" if comment_ratio > 0.40 else "under-documented"
        indicators.append(
            {
                "pattern": f"Abnormal Comment Density ({direction})",
                "severity": "medium",
                "description": f"Comment-to-code ratio is {comment_ratio:.0%}. "
                f"AI often {'adds excessive explanatory comments' if comment_ratio > 0.40 else 'omits inline comments entirely on long files'}.",
                "score": scores["comment_ratio"],
            }
        )

    # ---- HEURISTIC 18: Nested Loop / Conditional Depth ----
    # Humans write deeply nested code; AI avoids it for "clean code" style.
    max_nesting = 0
    current_nesting = 0
    nesting_depths = []
    for line in lines:
        stripped = line.strip()
        indent = len(line) - len(line.lstrip())
        # Count indent level as proxy for nesting (each 4 spaces = 1 level)
        indent_level = indent // 4
        nesting_depths.append(indent_level)
        if indent_level > max_nesting:
            max_nesting = indent_level

    avg_nesting = sum(nesting_depths) / len(nesting_depths) if nesting_depths else 0
    # If max nesting is very shallow for a file > 20 lines, score it
    shallow_nesting_score = 0
    if total_non_empty > 20:
        if max_nesting <= 2:
            shallow_nesting_score = min(80, (3 - max_nesting) * 35)
        elif avg_nesting < 1.0:
            shallow_nesting_score = min(50, (1.0 - avg_nesting) * 60)
    scores["shallow_nesting"] = round(shallow_nesting_score, 1)

    if shallow_nesting_score > 30:
        indicators.append(
            {
                "pattern": "Shallow Code Nesting",
                "severity": "low",
                "description": f"Maximum nesting depth is only {max_nesting} levels (avg {avg_nesting:.1f}) across {total_non_empty} lines. "
                f"AI strongly avoids deeply nested structures in favour of 'clean' flat code.",
                "score": scores["shallow_nesting"],
            }
        )

    # ---- HEURISTIC 19: Repeated Template Blocks ----
    # AI 'stitches' similar code blocks (slight variation on a template).
    # Detect repeated similar line patterns using n-gram repetition.
    stripped_lines = [l.strip() for l in lines if l.strip()]
    template_repeat_score = 0
    if len(stripped_lines) >= 10:
        # Build 3-line sliding windows and check for near-duplicates
        windows = []
        for i in range(len(stripped_lines) - 2):
            window = "\n".join(stripped_lines[i : i + 3])
            windows.append(window)
        # Count how many windows are very similar to a previous one
        seen = {}
        repeat_hits = 0
        for w in windows:
            # Normalize: remove digits and specific identifiers
            normalized = re.sub(r"\b\d+\b", "N", w)
            normalized = re.sub(r'"[^"]*"', "STR", normalized)
            normalized = re.sub(r"'[^']*'", "STR", normalized)
            if normalized in seen:
                repeat_hits += 1
            else:
                seen[normalized] = True
        repeat_ratio = repeat_hits / len(windows) if windows else 0
        template_repeat_score = min(100, repeat_ratio * 250)
    scores["template_repeat"] = round(template_repeat_score, 1)

    if template_repeat_score > 35:
        indicators.append(
            {
                "pattern": "Repeated Template Blocks",
                "severity": "high",
                "description": f"Detected structurally similar repeated code blocks (repeat ratio: {template_repeat_score:.0f}/100). "
                f"LLMs often 'stitch' slight variations of the same template pattern.",
                "score": scores["template_repeat"],
            }
        )

    # ---- HEURISTIC 20: Language-Specific Idiom Mismatch ----
    # AI uses specific modern-but-generic idioms regardless of context.
    # We look for a cluster of very common AI output markers per language.
    idiom_patterns = [
        # Python-specific AI idioms
        r'if\s+__name__\s*==\s*["\']__main__["\']',  # Every AI Python script ends with this
        r"logging\.basicConfig\(",  # AI always sets up logging formally
        r"argparse\.ArgumentParser\(",  # AI uses argparse even for simple scripts
        r'raise\s+ValueError\(["\']Invalid',  # AI's favourite exception phrasing
        # JS/TS-specific AI idioms
        r'console\.error\([`"\']Error:',  # AI error logging style
        r"Promise\.all\(",  # AI overuses Promise.all
        r"\.catch\(err\s*=>",  # AI's favourite catch param name
        r"process\.exit\(1\)",  # AI termination pattern
    ]
    idiom_hits = sum(1 for p in idiom_patterns if re.search(p, code))
    idiom_mismatch_score = min(100, idiom_hits * 20)
    scores["idiom_mismatch"] = round(idiom_mismatch_score, 1)

    if idiom_mismatch_score > 20:
        indicators.append(
            {
                "pattern": "AI Idiom Cluster",
                "severity": "high" if idiom_mismatch_score >= 60 else "medium",
                "description": f"Found {idiom_hits} language-specific AI output markers "
                f"(e.g., `if __name__ == '__main__'`, `logging.basicConfig`, `Promise.all`). "
                f"These often appear together in LLM-generated code.",
                "score": scores["idiom_mismatch"],
            }
        )

    # ---- HEURISTIC 21: Entropy & Predictability (Compression Ratio) ----
    # Replaces the 548MB GPT-2 perplexity model with a lightweight mathematical equivalent.
    # AI code is highly predictable and repetitive, leading to lower entropy and better compression.
    predictability_score = 0
    if len(code) > 100:
        compressed_size = len(zlib.compress(code.encode('utf-8')))
        ratio = compressed_size / len(code)
        
        # Lower ratio = better compression = highly predictable/repetitive (AI-like)
        # AI often sits around 0.25 - 0.35, Humans often 0.40+
        if ratio < 0.35:
            predictability_score = min(100, (0.35 - ratio) * 500)
            
    scores["perplexity"] = round(predictability_score, 1)

    if predictability_score > 30:
        indicators.append(
            {
                "pattern": "High Predictability (Low Entropy)",
                "severity": "high",
                "description": f"Code compresses abnormally well (ratio: {ratio:.2f}). AI generators tend to write highly predictable code with low statistical surprise.",
                "score": scores["perplexity"],
            }
        )

    # ---- HEURISTIC 22: Burstiness of Token Types ----
    # Calculate variance of keyword vs identifier vs literal frequencies
    tokens_alpha = re.findall(r"\b[a-zA-Z_]\w*\b", code)
    if len(tokens_alpha) > 10:
        keyword_count = sum(1 for t in tokens_alpha if t in LANGUAGE_KEYWORDS)
        identifier_count = len(tokens_alpha) - keyword_count
        literal_count = len(re.findall(r'["\'][^"\']*["\']|\b\d+\b', code))

        counts = [keyword_count, identifier_count, literal_count]
        mean_c = sum(counts) / 3
        variance_c = sum((c - mean_c) ** 2 for c in counts) / 3

        # AI often has a very balanced (low variance) token distribution
        burstiness_score = 0
        if variance_c < mean_c * 0.5:  # Extremely balanced
            burstiness_score = min(100, (mean_c * 0.5 - variance_c) * 5)
        scores["burstiness"] = round(burstiness_score, 1)
    else:
        scores["burstiness"] = 0

    if scores["burstiness"] > 30:
        indicators.append(
            {
                "pattern": "Low Token Burstiness",
                "severity": "medium",
                "description": "Token type distribution (keywords vs identifiers vs literals) is unusually uniform, a common artifact of LLM generation.",
                "score": scores["burstiness"],
            }
        )

    # ---- HEURISTIC 23: Repeated API-Call Patterns ----
    # Match object.method() patterns
    api_calls = re.findall(r"\b([a-zA-Z_]\w*\.[a-zA-Z_]\w*)\s*\(", code)
    api_score = 0
    if len(api_calls) > 5:
        # Check for sequential n-grams of API calls
        api_ngrams = []
        for i in range(len(api_calls) - 2):
            api_ngrams.append(f"{api_calls[i]}|{api_calls[i + 1]}|{api_calls[i + 2]}")

        if api_ngrams:
            api_counts = Counter(api_ngrams)
            most_common = api_counts.most_common(1)[0]
            if most_common[1] >= 3:  # Repeated the exact same 3 API calls >= 3 times
                api_score = min(100, most_common[1] * 20)
    scores["api_pattern"] = round(api_score, 1)

    if api_score > 30:
        indicators.append(
            {
                "pattern": "Repeated API Call Sequences",
                "severity": "high",
                "description": f"Found highly repetitive API call sequences. AI models often repeatedly use the exact same chain of method calls.",
                "score": scores["api_pattern"],
            }
        )

    # ---- HEURISTIC 24: Statistical N-gram Entropy ----
    # Compute Shannon entropy of character trigrams
    trigrams = [code[i : i + 3] for i in range(len(code) - 2)]
    ngram_score = 0
    if len(trigrams) > 50:
        trigram_counts = Counter(trigrams)
        total_trigrams = len(trigrams)
        entropy = -sum(
            (count / total_trigrams) * math.log2(count / total_trigrams)
            for count in trigram_counts.values()
        )
        # Lower entropy means more predictable
        if entropy < 4.0:
            ngram_score = min(100, (4.0 - entropy) * 40)
    scores["ngram_entropy"] = round(ngram_score, 1)

    if ngram_score > 30:
        indicators.append(
            {
                "pattern": "Low N-gram Entropy",
                "severity": "medium",
                "description": f"Character-level trigram entropy is unusually low. AI generated text often has smoother, less complex n-gram distributions.",
                "score": scores["ngram_entropy"],
            }
        )

    # ---- HEURISTIC 25: Stylistic Drift Detection ----
    # Measure line length variance in blocks
    drift_score = 0
    if len(lines) > 20:
        block_size = len(lines) // 4
        block_variances = []
        for i in range(4):
            block = lines[i * block_size : (i + 1) * block_size]
            lengths = [len(l.strip()) for l in block if l.strip()]
            if lengths:
                mean_l = sum(lengths) / len(lengths)
                var_l = sum((l - mean_l) ** 2 for l in lengths) / len(lengths)
                block_variances.append(var_l)

        if len(block_variances) == 4:
            # How much does the variance change across blocks?
            mean_var = sum(block_variances) / 4
            drift = sum((v - mean_var) ** 2 for v in block_variances) / 4
            if mean_var > 0:
                drift_ratio = drift / mean_var
                if drift_ratio < 10:  # Very low drift = high uniformity
                    drift_score = min(100, (10 - drift_ratio) * 10)
    scores["style_drift"] = round(drift_score, 1)

    if drift_score > 30:
        indicators.append(
            {
                "pattern": "Minimal Stylistic Drift",
                "severity": "medium",
                "description": "Code style (e.g., line lengths) remains extremely consistent throughout the file. Human authors usually exhibit stylistic 'drift' over long files.",
                "score": scores["style_drift"],
            }
        )

    # ============================================================
    # CALCULATE OVERALL AI PROBABILITY
    # ============================================================

    # Weighted average of all 25 heuristic scores (sum = 1.00)
    weights = {
        "commenting": 0.06,
        "naming": 0.11,
        "formatting": 0.09,
        "boilerplate": 0.06,
        "documentation": 0.05,
        "sequential_steps": 0.02,
        "repetitive_structure": 0.05,
        "error_handling": 0.02,
        "cleanliness": 0.09,
        "imports": 0.03,
        "type_hints": 0.03,
        "immutability_bias": 0.02,
        "return_verbosity": 0.01,
        "over_modularization": 0.02,
        "outdated_idioms": 0.01,
        "name_entropy": 0.03,
        "comment_ratio": 0.02,
        "shallow_nesting": 0.01,
        "template_repeat": 0.06,
        "idiom_mismatch": 0.02,
        "perplexity": 0.06,
        "burstiness": 0.03,
        "api_pattern": 0.06,
        "ngram_entropy": 0.02,
        "style_drift": 0.02,
    }

    ai_probability = 0
    for key, weight in weights.items():
        ai_probability += scores.get(key, 0) * weight

    ai_probability = round(min(100, max(0, ai_probability)), 1)

    # Determine verdict based on STRICT_MODE

    if ai_probability >= 75:
        verdict = "Highly Likely AI-Generated"
    elif ai_probability >= 50:
        verdict = "Likely AI-Generated"
    elif ai_probability >= 30:
        verdict = "Possibly AI-Assisted"
    elif ai_probability >= 15:
        verdict = "Likely Human-Written"
    else:
        verdict = "Human-Written"

    # Sort indicators by score (highest first)
    indicators.sort(key=lambda x: x["score"], reverse=True)

    return {
        "ai_probability": ai_probability,
        "verdict": verdict,
        "indicators": indicators,
        "details": scores,
        "summary": {
            "total_lines": total_lines,
            "code_lines": total_non_empty,
            "comment_lines": len(comment_lines),
            "functions_found": func_count,
            "indicators_triggered": len(indicators),
        },
    }


# ============================================================
# API ENDPOINTS
# ============================================================


@app.route("/detect-ai", methods=["POST"])
def detect_ai():
    """
    POST /detect-ai

    Accepts a single code string and analyzes it for AI-generated
    code patterns using 10 different heuristic checks.

    Request body (JSON):
    {
        "code": "function hello() { ... }"
    }
    """
    data = request.get_json()

    if not data or "code" not in data:
        return jsonify({"error": "Please provide 'code' in the request body."}), 400

    code = data["code"]

    if not code.strip():
        return jsonify(
            {
                "ai_probability": 0,
                "verdict": "Unable to analyze",
                "indicators": [],
                "details": {},
                "summary": {},
            }
        )

    result = detect_ai_patterns(code)
    return jsonify(result)


@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint."""
    return jsonify(
        {
            "status": "ok",
            "service": "Python AI Microservice (Advanced)",
            "algorithms": [
                "TF-IDF Cosine (code-aware)",
                "N-gram Winnowing Fingerprint",
                "Structural Line Matching",
                "Pattern Analysis",
                "Transformer Embedding Similarity",
                "ML Calibrator (optional)",
                "AI Code Detection (25 heuristics)",
            ],
        }
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    print("\n[*] Advanced AI Plagiarism Microservice starting...")
    print("[>] Endpoints:")
    print("    POST http://localhost:8000/analyze")
    print("    POST http://localhost:8000/train-calibrator")
    print("    POST http://localhost:8000/detect-ai")
    print("    GET  http://localhost:8000/model-info")
    print("    GET  http://localhost:8000/health")
    print("\n[+] Algorithms loaded:")
    print("    - TF-IDF Cosine Similarity (code-aware)")
    print("    - N-gram Winnowing Fingerprint (structural matching)")
    print("    - AST-aware Sequence Matching (logic-level diff)")
    print("    - Code Pattern Analysis")
    print("    - Transformer Embedding Similarity (CodeBERT)")
    print("    - ML Calibrator (optional)")
    print("    - AI Code Detection (25 heuristics)\n")
    app.run(host="0.0.0.0", port=8000, debug=True)
