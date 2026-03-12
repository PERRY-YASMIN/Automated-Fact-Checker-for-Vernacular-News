from __future__ import annotations

import json
from pathlib import Path
from typing import List

from ml import config
from ml.inference.pipeline import fact_check_text


def load_sample_posts(path: Path | None = None) -> List[dict]:
    if path is None:
        path = config.SAMPLE_POSTS_PATH
    posts = []
    for line in path.read_text(encoding="utf-8").splitlines():
        posts.append(json.loads(line))
    return posts


def process_post(post: dict) -> None:
    post_id = post["id"]
    text = post["text"]
    result = fact_check_text(text)

    print("=" * 80)
    print(f"Post {post_id}")
    print(f"Original text: {text}")
    print(f"Extracted {len(result['claims'])} potential claim(s):")
    for entry in result["claims"]:
        print(f"  - {entry['claim']}")
        print(f"    -> Verdict: {entry['verdict']} (conf={entry['confidence']:.3f})")
        if not entry["evidence"]:
            print("    -> No matching facts above similarity threshold.")
            continue
        print("    -> Top matching facts:")
        for evidence in entry["evidence"]:
            print(f"       * {evidence['id']} ({evidence['score']:.3f}): {evidence['claim']}")


def main() -> None:
    # Windows terminals may default to a non-UTF8 code page (e.g., cp1252),
    # which can crash prints for Indic scripts. Force UTF-8 output.
    import sys

    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    posts = load_sample_posts()
    for post in posts:
        process_post(post)


if __name__ == "__main__":
    main()