from __future__ import annotations

from ml.inference.retrieval_pipeline import rebuild_fact_index
from ml.pipeline import verify_claim


def run_demo() -> None:
    # Rebuild retrieval index to reflect any newly added facts.
    rebuild_fact_index()

    demo_claims = [
        "The Earth is the third planet from the Sun.",
        "पृथ्वी सूर्य से तीसरे नंबर का ग्रह है।",
        "500 और 1000 के नोट वापस शुरू कर दिए गए हैं।",
    ]

    print("=" * 80)
    print("Vernacular Fact-Checker Pipeline Demo")
    print("=" * 80)
    for claim in demo_claims:
        result = verify_claim(claim)
        print(f"Claim: {claim}")
        print(f"Verdict: {result['verdict']} | Confidence: {result['confidence']:.3f}")
        if result["sources"]:
            print("Sources:")
            for src in result["sources"][:3]:
                print(f"  - {src}")
        else:
            print("Sources: []")
        print("-" * 80)


def main() -> None:
    # Windows terminals may default to a non-UTF8 code page (e.g., cp1252),
    # which can crash prints for Indic scripts. Force UTF-8 output.
    import sys

    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    run_demo()


if __name__ == "__main__":
    main()