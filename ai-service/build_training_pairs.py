import argparse
import json
import os
import random
from pathlib import Path


def _parse_extensions(raw):
    if not raw:
        return set()
    return {ext.strip().lower() for ext in raw.split(",") if ext.strip()}


def _collect_files(root, extensions):
    files = []
    for path in Path(root).rglob("*"):
        if not path.is_file():
            continue
        if extensions and path.suffix.lower() not in extensions:
            continue
        files.append(path)
    return files


def _read_text(path, max_chars):
    text = path.read_text(encoding="utf-8", errors="ignore")
    if max_chars and len(text) > max_chars:
        return text[:max_chars]
    return text


def build_pairs(original_dir, plagiarized_dir, positive_limit, negative_limit, seed, extensions, max_chars):
    random.seed(seed)

    original_dir = Path(original_dir)
    plagiarized_dir = Path(plagiarized_dir)

    orig_files = _collect_files(original_dir, extensions)
    plag_files = _collect_files(plagiarized_dir, extensions)

    orig_map = {str(path.relative_to(original_dir)): path for path in orig_files}
    plag_map = {str(path.relative_to(plagiarized_dir)): path for path in plag_files}

    shared_keys = sorted(set(orig_map.keys()) & set(plag_map.keys()))
    positive_pairs = []

    for key in shared_keys:
        if positive_limit and len(positive_pairs) >= positive_limit:
            break
        code1 = _read_text(orig_map[key], max_chars)
        code2 = _read_text(plag_map[key], max_chars)
        positive_pairs.append(
            {
                "name1": key,
                "name2": key,
                "code1": code1,
                "code2": code2,
                "label": 1,
            }
        )

    negative_pairs = []
    if orig_files and plag_files:
        attempts = 0
        max_attempts = max(negative_limit * 20, 200)
        seen = set()
        while len(negative_pairs) < negative_limit and attempts < max_attempts:
            attempts += 1
            left = random.choice(orig_files)
            right = random.choice(plag_files)
            left_key = str(left.relative_to(original_dir))
            right_key = str(right.relative_to(plagiarized_dir))
            if left_key == right_key:
                continue
            pair_key = (left_key, right_key)
            if pair_key in seen:
                continue
            seen.add(pair_key)
            negative_pairs.append(
                {
                    "name1": left_key,
                    "name2": right_key,
                    "code1": _read_text(left, max_chars),
                    "code2": _read_text(right, max_chars),
                    "label": 0,
                }
            )

    pairs = positive_pairs + negative_pairs
    random.shuffle(pairs)
    return {
        "pairs": pairs,
        "summary": {
            "positives": len(positive_pairs),
            "negatives": len(negative_pairs),
            "total": len(pairs),
        },
    }


def main():
    parser = argparse.ArgumentParser(
        description="Build labeled training pairs from two folder trees."
    )
    parser.add_argument("--original-dir", required=True, help="Folder with original code")
    parser.add_argument(
        "--plagiarized-dir", required=True, help="Folder with plagiarized code"
    )
    parser.add_argument("--output", required=True, help="Output JSON path")
    parser.add_argument(
        "--positive", type=int, default=200, help="Max positive pairs to include"
    )
    parser.add_argument(
        "--negative", type=int, default=200, help="Max negative pairs to include"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--extensions",
        default=".py,.js,.ts,.java,.cpp,.c",
        help="Comma-separated extensions",
    )
    parser.add_argument(
        "--max-chars",
        type=int,
        default=20000,
        help="Max chars per file (0 = unlimited)",
    )

    args = parser.parse_args()
    extensions = _parse_extensions(args.extensions)
    max_chars = args.max_chars if args.max_chars > 0 else None

    data = build_pairs(
        args.original_dir,
        args.plagiarized_dir,
        args.positive,
        args.negative,
        args.seed,
        extensions,
        max_chars,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2)

    summary = data["summary"]
    print(
        "Wrote {total} pairs ({positives} positive, {negatives} negative) to {path}".format(
            total=summary["total"],
            positives=summary["positives"],
            negatives=summary["negatives"],
            path=output_path,
        )
    )


if __name__ == "__main__":
    main()
