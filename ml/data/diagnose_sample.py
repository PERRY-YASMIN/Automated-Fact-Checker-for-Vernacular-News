#!/usr/bin/env python3
"""Diagnose retrieval and verification for specific sample."""
import json
import requests

sample_id = "sample2"
BASE_URL = "http://127.0.0.1:8000/verify"

# Load sample
from pathlib import Path
sample_file = Path(__file__).parent / "verify_test_samples.jsonl"
with open(sample_file, encoding="utf-8") as f:
    samples = {json.loads(line)["id"]: json.loads(line) for line in f}

s = samples[sample_id]
print(f"Sample: {sample_id}")
print(f"Text: {s['text']}")
print(f"Expected: {s.get('expected_verdict', '?')}")
print()

# Call verify endpoint
r = requests.post(BASE_URL, json={"text": s["text"]}, timeout=60)
result = r.json()

print(f"Actual verdict: {result.get('verdict', 'ERROR')}")
print(f"Confidence: {result.get('confidence', 0.0):.4f}")
print()
print("Retrieved facts:")
for src in result.get("sources", []):
    print(f"  {src['id']:<10} score={src['score']:.4f}  {src['claim'][:70]}")
