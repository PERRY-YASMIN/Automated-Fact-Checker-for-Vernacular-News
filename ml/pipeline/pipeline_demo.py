from __future__ import annotations

import json
from pathlib import Path
from typing import List

from ml import config
from .claim_extraction import extract_claims_from_post, Claim
from .language_id import detect_language
from .retrieval import load_facts, build_fact_index, retrieve_for_claim, RetrievedFact


def load_sample_posts(path: Path | None = None) -> List[dict]:
    if path is None:
        path = config.SAMPLE_POSTS_PATH
    posts = []
    for line in path.read_text(encoding="utf-8").splitlines():
        posts.append(json.loads(line))
    return posts


def process_post(
    post: dict,
    fact_embeddings,
    facts,
) -> None:
    post_id = post["id"]
    text = post["text"]
    lang_info = detect_language(text)
    claims: List[Claim] = extract_claims_from_post(post_id, text)

    print("=" * 80)
    print(f"Post {post_id}")
    print(f"Original text: {text}")
    print(f"Detected language: {lang_info.lang} (score={lang_info.score:.3f}, script={lang_info.script})")
    print(f"Extracted {len(claims)} potential claim(s):")
    for c in claims:
        print(f"  - [{c.sentence_index}] {c.text}")

        retrieved: List[RetrievedFact] = retrieve_for_claim(c.text, fact_embeddings, facts)
        if not retrieved:
            print("    -> No matching facts above similarity threshold.")
            continue
        print("    -> Top matching facts:")
        for r in retrieved:
            print(f"       * {r.fact.id} ({r.score:.3f}): {r.fact.claim}")


def main() -> None:
    posts = load_sample_posts()
    facts = load_facts()
    fact_embeddings, facts = build_fact_index(facts)

    for post in posts:
        process_post(post, fact_embeddings, facts)


if __name__ == "__main__":
    main()