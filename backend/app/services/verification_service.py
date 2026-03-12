import wikipedia


def verify_claim_logic(claim_text: str):

    try:
        # search wikipedia
        results = wikipedia.search(claim_text)

        if not results:
            return {
                "verdict": "Unverified",
                "confidence": 0.4,
                "evidence": "No Wikipedia evidence found."
            }

        # get summary of first result
        summary = wikipedia.summary(results[0], sentences=2)

        return {
            "verdict": "Needs Review",
            "confidence": 0.6,
            "evidence": summary
        }

    except Exception:
        return {
            "verdict": "Unverified",
            "confidence": 0.3,
            "evidence": "Error retrieving evidence."
        }