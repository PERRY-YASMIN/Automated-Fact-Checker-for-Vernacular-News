#!/usr/bin/env python3
"""Smoke test: Verify 8 critical samples against the /verify endpoint."""
import json
import requests
from pathlib import Path

BASE_URL = "http://127.0.0.1:8000/verify"

# Core test samples covering all verdict types
TEST_SAMPLES = ["sample1", "sample3", "sample11", "sample12", "sample13", "sample17", "sample21", "sample24"]

# Load all samples from JSONL
sample_file = Path(__file__).parent / "verify_test_samples.jsonl"
samples = {}
with open(sample_file, encoding="utf-8") as f:
    for line in f:
        s = json.loads(line)
        samples[s["id"]] = s

# Run smoke test
results = []
print("Testing 8 samples against http://127.0.0.1:8000/verify\n")
print(f"{'ID':<12} {'EXPECTED':<25} {'ACTUAL':<25} {'CONF':>8} {'PASS':<8}")
print("-" * 80)

for sample_id in TEST_SAMPLES:
    if sample_id not in samples:
        print(f"{sample_id:<12} ERROR: Sample not found")
        continue
    
    s = samples[sample_id]
    try:
        r = requests.post(BASE_URL, json={"text": s["text"]}, timeout=60)
        result = r.json()
        expected = s.get("expected_verdict", "?")
        actual = result.get("verdict", "ERROR")
        conf = result.get("confidence", 0.0)
        passed = expected == actual
        results.append(passed)
        
        status = "YES" if passed else "NO"
        print(
            f"{sample_id:<12} {expected:<25} {actual:<25} "
            f"{conf:>7.3f}  {status:<8}"
        )
    except Exception as e:
        print(f"{sample_id:<12} ERROR: {str(e)[:50]}")
        results.append(False)

print("-" * 80)
passed_count = sum(results)
total_count = len(results)
print(f"Result: {passed_count}/{total_count} passed")
