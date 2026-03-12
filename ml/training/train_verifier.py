from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from ml import config


LABEL_TO_ID = {
    "contradiction": 0,
    "neutral": 1,
    "entailment": 2,
    "refuted": 0,
    "notenoughevidence": 1,
    "not enough evidence": 1,
    "supported": 2,
}


def _read_jsonl(path: Path) -> pd.DataFrame:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return pd.DataFrame(rows)


def load_verifier_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    if path.suffix.lower() == ".jsonl":
        df = _read_jsonl(path)
    else:
        df = pd.read_csv(path)

    premise_col = "premise" if "premise" in df.columns else "fact"
    hypothesis_col = "hypothesis" if "hypothesis" in df.columns else "claim"
    if premise_col not in df.columns or hypothesis_col not in df.columns:
        raise ValueError("Dataset must include premise/fact and hypothesis/claim columns")
    if "label" not in df.columns:
        raise ValueError("Dataset must include a label column")

    out = df[[premise_col, hypothesis_col, "label"]].rename(
        columns={premise_col: "premise", hypothesis_col: "hypothesis"}
    )
    out["premise"] = out["premise"].astype(str)
    out["hypothesis"] = out["hypothesis"].astype(str)
    out["label"] = out["label"].astype(str).str.lower().map(LABEL_TO_ID)
    out = out.dropna(subset=["label"]) 
    out["label"] = out["label"].astype(int)
    return out


@dataclass
class NliItem:
    premise: str
    hypothesis: str
    label: int


class NliDataset(Dataset):
    def __init__(self, rows: List[NliItem], tokenizer, max_length: int = 256):
        self.rows = rows
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.rows[idx]
        enc = self.tokenizer(
            item.premise,
            item.hypothesis,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        out = {k: v.squeeze(0) for k, v in enc.items()}
        out["labels"] = torch.tensor(item.label, dtype=torch.long)
        return out


def train_verifier(data: pd.DataFrame, output_dir: Path, epochs: int, batch_size: int, lr: float) -> None:
    rows = [NliItem(r.premise, r.hypothesis, int(r.label)) for r in data.itertuples(index=False)]
    if len(rows) < 10:
        raise ValueError("Need at least 10 labeled NLI samples for verifier training")

    split = int(0.9 * len(rows))
    train_rows = rows[:split]
    valid_rows = rows[split:]

    tokenizer = AutoTokenizer.from_pretrained(config.VERIFIER_MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(config.VERIFIER_MODEL_NAME, num_labels=3)

    device = torch.device(config.get_device())
    model.to(device)

    train_ds = NliDataset(train_rows, tokenizer)
    valid_ds = NliDataset(valid_rows, tokenizer)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_ds, batch_size=batch_size, shuffle=False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(**batch)
            loss = out.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            total_loss += float(loss.item())

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in valid_loader:
                labels = batch["labels"].to(device)
                inputs = {k: v.to(device) for k, v in batch.items() if k != "labels"}
                logits = model(**inputs).logits
                preds = torch.argmax(logits, dim=-1)
                correct += int((preds == labels).sum().item())
                total += int(labels.numel())

        avg_loss = total_loss / max(len(train_loader), 1)
        val_acc = correct / max(total, 1)
        print(f"Epoch {epoch}/{epochs} - train_loss={avg_loss:.4f} val_acc={val_acc:.4f}")

    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(output_dir)
    model.save_pretrained(output_dir)

    metadata = {
        "base_model": config.VERIFIER_MODEL_NAME,
        "samples": len(rows),
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": lr,
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved verifier model to: {output_dir}")


def generate_nli_from_kb(kb_path: Path) -> pd.DataFrame:
    """Auto-generate multilingual NLI pairs from verified_facts.jsonl.

    Pairing strategy (using topic_id field):
    - ENTAILMENT  : any two facts that share the same topic_id (bilingual
                    translations, or same-language paraphrases).  In a 200-
                    fact KB with ~100 bilingual pairs this produces ~100 EN<->HI
                    cross-lingual entailment examples \u2014 the most valuable pairs
                    for cross-lingual generalisation.
    - NEUTRAL     : facts from *different* topic_ids.  The claims are about
                    unrelated subjects so the NLI label is neutral.
    - CONTRADICTION (proxy): a fact paired with a randomly constructed negative
                    by prepending \"It is false that\" to a different-topic claim.

    The generated dataset is small (a few hundred rows) and is intended for
    fine-tuning the NLI head on top of the pre-trained xlm-roberta model.
    For larger-scale training, supply your own labelled dataset via --data-path.
    """
    facts: List[Dict] = []
    for line in kb_path.read_text(encoding=\"utf-8\").splitlines():
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        if obj.get(\"claim\", \"\").strip():
            facts.append(obj)

    # Group by topic_id
    by_topic: Dict[str, List[Dict]] = {}
    for f in facts:
        tid = str(f.get(\"topic_id\", f[\"id\"]))
        by_topic.setdefault(tid, []).append(f)

    rows: List[Dict] = []
    random.seed(42)
    topic_ids = list(by_topic.keys())

    # --- ENTAILMENT: same-topic pairs (includes cross-lingual bilingual pairs) ---
    for tid, group in by_topic.items():
        if len(group) < 2:
            continue
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                rows.append({
                    \"premise\": group[i][\"claim\"],
                    \"hypothesis\": group[j][\"claim\"],
                    \"label\": \"entailment\",
                })

    # --- NEUTRAL: cross-topic random pairs ---
    n_neutral = max(len(rows), 50)
    for _ in range(n_neutral):
        t1, t2 = random.sample(topic_ids, 2)
        f1 = random.choice(by_topic[t1])
        f2 = random.choice(by_topic[t2])
        rows.append({
            \"premise\": f1[\"claim\"],
            \"hypothesis\": f2[\"claim\"],
            \"label\": \"neutral\",
        })

    # --- CONTRADICTION (proxy): \"It is false that <different-topic claim>\" ---
    n_contra = max(len(rows) // 3, 20)
    for _ in range(n_contra):
        t1, t2 = random.sample(topic_ids, 2)
        f_premise = random.choice(by_topic[t1])
        f_other = random.choice(by_topic[t2])
        negated = \"It is false that \" + f_other[\"claim\"].rstrip(\".\") + \".\"
        rows.append({
            \"premise\": f_premise[\"claim\"],
            \"hypothesis\": negated,
            \"label\": \"contradiction\",
        })

    df = pd.DataFrame(rows).sample(frac=1, random_state=42).reset_index(drop=True)
    label_counts = df[\"label\"].value_counts().to_dict()
    print(
        f\"Generated {len(df)} NLI pairs from KB ({kb_path}): \"\n        + \", \".join(f\"{k}={v}\" for k, v in label_counts.items())
    )
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description=\"Train multilingual verifier model and persist artifacts.\")
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument(
        \"--data-path\",
        type=str,
        help=\"Path to CSV/JSONL with premise/hypothesis/label columns.\",
    )
    source_group.add_argument(
        \"--generate-from-kb\",
        action=\"store_true\",
        help=(
            \"Auto-generate NLI training pairs from the verified facts KB \"\n            \"(ml/data/verified_facts.jsonl).  Cross-lingual bilingual pairs \"\n            \"sharing the same topic_id become entailment examples; random cross-\"\n            \"topic pairs become neutral; negated claims become contradiction.\"\n        ),
    )
    parser.add_argument(
        \"--kb-path\",
        type=str,
        default=str(config.VERIFIED_FACTS_PATH),
        help=\"Path to verified_facts.jsonl (used only with --generate-from-kb).\",
    )
    parser.add_argument(
        \"--output-dir\",
        type=str,
        default=str(config.VERIFIER_MODEL_DIR),
        help=\"Directory to store trained verifier model.\",
    )
    parser.add_argument(\"--epochs\", type=int, default=2, help=\"Training epochs\")
    parser.add_argument(\"--batch-size\", type=int, default=8, help=\"Batch size\")
    parser.add_argument(\"--learning-rate\", type=float, default=2e-5, help=\"Learning rate\")
    args = parser.parse_args()

    if args.generate_from_kb:
        data = generate_nli_from_kb(Path(args.kb_path))
    else:
        data = load_verifier_data(Path(args.data_path))

    train_verifier(data, Path(args.output_dir), args.epochs, args.batch_size, args.learning_rate)


if __name__ == \"__main__\":
    main()
