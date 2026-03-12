from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import List

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from ml import config


def _read_jsonl(path: Path) -> pd.DataFrame:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return pd.DataFrame(rows)


def load_training_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    if path.suffix.lower() == ".jsonl":
        df = _read_jsonl(path)
    else:
        df = pd.read_csv(path)

    # Flexible column support: text/claim_text/sentence + label/is_claim.
    text_col = None
    for col in ("text", "claim_text", "sentence"):
        if col in df.columns:
            text_col = col
            break
    if text_col is None:
        raise ValueError("Dataset must include one of: text, claim_text, sentence")

    label_col = None
    for col in ("label", "is_claim"):
        if col in df.columns:
            label_col = col
            break
    if label_col is None:
        raise ValueError("Dataset must include one of: label, is_claim")

    out = df[[text_col, label_col]].rename(columns={text_col: "text", label_col: "label"})
    out["text"] = out["text"].astype(str)
    out["label"] = out["label"].astype(int)
    return out


def train_and_save(df: pd.DataFrame, output_dir: Path) -> None:
    x_train, x_test, y_train, y_test = train_test_split(
        df["text"],
        df["label"],
        test_size=0.2,
        random_state=42,
        stratify=df["label"],
    )

    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=50000,
        lowercase=True,
    )
    x_train_vec = vectorizer.fit_transform(x_train)
    x_test_vec = vectorizer.transform(x_test)

    model = LogisticRegression(max_iter=1000, class_weight="balanced")
    model.fit(x_train_vec, y_train)

    y_pred = model.predict(x_test_vec)
    print("Claim Detector Evaluation")
    print(classification_report(y_test, y_pred, digits=4))

    output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output_dir / "model.joblib")
    joblib.dump(vectorizer, output_dir / "vectorizer.joblib")

    metadata = {
        "model_type": "tfidf_logreg",
        "samples": int(len(df)),
        "features": int(x_train_vec.shape[1]),
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved claim detector artifacts to: {output_dir}")


# ── Non-claim text patterns used when --generate-from-kb is active ────────────
_NON_CLAIM_TEMPLATES: List[str] = [
    "Please share this message with everyone you know!",
    "Good morning! Have a great day.",
    "Happy birthday! Wishing you all the best.",
    "Like and subscribe to my channel for more updates.",
    "Forward this to as many people as possible.",
    "Congratulations on your success!",
    "Thanks for watching. See you next time!",
    "Follow me for more content.",
    "इसे ज्यादा से ज्यादा लोगों तक पहुँचाएं।",
    "सुप्रभात! आपका दिन शुभ हो।",
    "जन्मदिन की हार्दिक शुभकामनाएँ।",
    "इस पोस्ट को शेयर करें।",
    "नमस्ते दोस्तों, कैसे हैं आप?",
    "Click here for the latest deals and offers.",
    "Subscribe now and never miss an update.",
    "Check out this amazing product!",
    "यह वीडियो देखना मत भूलिए।",
    "आज का मौसम बहुत अच्छा है।",
    "Have a wonderful evening!",
    "Thanks for your support. God bless you.",
    "Greetings from our team to yours!",
    "यह जानकारी सभी के साथ साझा करें।",
    "Breaking news! Watch this video now.",
    "LOL this is hilarious!",
    "OMG! You won't believe this!",
    "😂😂😂 So funny!",
    "Please pray for us.",
    "Urgent: Forward to all contacts immediately!",
    "Wishing everyone a safe and happy holiday.",
    "Stay tuned for more updates.",
]


def generate_from_kb(kb_path: Path) -> pd.DataFrame:
    """Auto-generate a labelled claim-detection dataset from verified_facts.jsonl.

    Strategy:
    - ALL fact claims in the KB  -> label 1  (is a claim)
    - Synthetic social/conversational sentences -> label 0  (not a claim)

    The synthetic negatives are drawn from _NON_CLAIM_TEMPLATES and
    optionally augmented with partial-sentence fragments clipped from
    the KB claims (words only, no verifiable assertion).
    """
    facts: List[str] = []
    for line in kb_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        claim_text = str(obj.get("claim", "")).strip()
        if claim_text:
            facts.append(claim_text)

    positives = pd.DataFrame({"text": facts, "label": 1})

    # Build negatives: fixed templates + short fragments from fact claims (no
    # full assertion) so the model learns to reject incomplete snippets.
    negatives_raw = list(_NON_CLAIM_TEMPLATES)
    random.seed(42)
    for fact in random.sample(facts, min(len(facts), 30)):
        words = fact.split()
        if len(words) >= 4:
            # Take first 2-3 words only — no predicate, therefore not a claim.
            snippet = " ".join(words[: random.randint(2, 3)])
            negatives_raw.append(snippet)

    # Ensure at least 1 negative per positive for balanced training.
    while len(negatives_raw) < len(facts):
        negatives_raw.extend(_NON_CLAIM_TEMPLATES)
    negatives_raw = negatives_raw[: len(facts)]

    negatives = pd.DataFrame({"text": negatives_raw, "label": 0})
    df = pd.concat([positives, negatives], ignore_index=True).sample(
        frac=1, random_state=42
    ).reset_index(drop=True)

    print(
        f"Generated {len(positives)} positive and {len(negatives)} negative examples "
        f"from KB ({kb_path})"
    )
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Train sentence-level claim detector and persist artifacts.")
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument(
        "--data-path",
        type=str,
        help="Path to CSV/JSONL with text+label columns.",
    )
    source_group.add_argument(
        "--generate-from-kb",
        action="store_true",
        help=(
            "Auto-generate training data from the verified facts KB "
            "(ml/data/verified_facts.jsonl).  All KB claims become positive "
            "examples; synthetic conversational sentences become negatives."
        ),
    )
    parser.add_argument(
        "--kb-path",
        type=str,
        default=str(config.VERIFIED_FACTS_PATH),
        help="Path to verified_facts.jsonl (used only with --generate-from-kb).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(config.CLAIM_DETECTOR_MODEL_DIR),
        help="Directory to store trained claim detector artifacts.",
    )
    args = parser.parse_args()

    if args.generate_from_kb:
        df = generate_from_kb(Path(args.kb_path))
    else:
        df = load_training_data(Path(args.data_path))

    train_and_save(df, Path(args.output_dir))


if __name__ == "__main__":
    main()
