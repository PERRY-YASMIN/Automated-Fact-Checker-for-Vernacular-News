#!/usr/bin/env python3
"""Test all 25 verify samples and report comprehensive results."""
import json
import requests
import sys

BASE_URL = "http://127.0.0.1:8000/verify"

# Load samples
from pathlib import Path
sample_file = Path(__file__).parent / "verify_test_samples.jsonl"
with open(sample_file, encoding="utf-8") as f:
    samples = [json.loads(line) for line in f]

results = []
for s in samples:
    try:
        r = requests.post(BASE_URL, json={"text": s["text"]}, timeout=60)
        result = r.json()
        expected = s.get("expected_verdict", "?")
        actual = result.get("verdict", "ERROR")
        conf = result.get("confidence", 0.0)
        passed = expected == actual
        results.append({
            "id": s["id"],
            "expected": expected,
            "actual": actual,
            "conf": conf,
            "passed": passed,
            "language": s.get("language", "?"),
        })
    except Exception as e:
        results.append({"id": s["id"], "error": str(e), "passed": False})

# Print results table
print(f"{'ID':<10} {'LANG':<5} {'EXPECTED':<20} {'ACTUAL':<20} {'CONF':>6} {'PASS':<5}")
print("-" * 88)

for r in results:
    if "error" in r:
        print(f"{r['id']:<10} ERROR: {r['error'][:50]}")
    else:
        status = "✓" if r["passed"] else "✗"
        lang = r.get("language", "?")[:4]
        print(
            f"{r['id']:<10} {lang:<5} {r['expected']:<20} {r['actual']:<20} "
            f"{r['conf']:>6.3f} {status:<5}"
        )

passed = sum(1 for r in results if r.get("passed", False))
print("-" * 88)
print(f"Total: {passed}/{len(results)} passed ({100*passed/len(results):.1f}%)\n")

# Breakdown by verdict
by_verdict = {}
for r in results:
    if not r.get("error"):
        v = r["expected"]
        if v not in by_verdict:
            by_verdict[v] = {"total": 0, "passed": 0}
        by_verdict[v]["total"] += 1
        if r["passed"]:
            by_verdict[v]["passed"] += 1

print("Breakdown by verdict type:")
for v in ["Supported", "Refuted", "NotEnoughEvidence"]:
    if v in by_verdict:
        d = by_verdict[v]
        pct = 100 * d["passed"] / d["total"]
        print(f"  {v:<20} {d['passed']}/{d['total']} ({pct:.0f}%)")

sys.exit(0 if passed == len(results) else 1)
