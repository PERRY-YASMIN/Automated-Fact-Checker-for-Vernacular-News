from ml.inference.pipeline import fact_check_text


def test_fact_check_text_returns_claims_and_verdict(monkeypatch):
    monkeypatch.setattr("ml.inference.pipeline.embed_text", lambda _text: [0.1, 0.2, 0.3])

    class _Fact:
        def __init__(self, fact_id: str, claim: str, language: str = "en"):
            self.id = fact_id
            self.claim = claim
            self.language = language

    class _Retrieved:
        def __init__(self):
            self.fact = _Fact("fact-demo", "Petrol prices are not 120 in Delhi.")
            self.score = 0.88

    monkeypatch.setattr("ml.inference.pipeline.retrieve_facts", lambda _claim, k=5: [_Retrieved()])
    monkeypatch.setattr(
        "ml.inference.pipeline.verify_claim",
        lambda _claim, _facts: {
            "verdict": "Refuted",
            "confidence": 0.91,
            "best_fact_id": "fact-demo",
            "best_fact_score": 0.88,
            "nli_probs": {"contradiction": 0.91, "neutral": 0.06, "entailment": 0.03},
        },
    )

    text = "आज पेट्रोल की कीमत 120 रुपये प्रति लीटर हो गई है. कृपया सबको बताएं."
    result = fact_check_text(text)

    assert "claims" in result
    assert len(result["claims"]) >= 1
    assert result["claims"][0]["verdict"] in {"Supported", "Refuted", "NotEnoughEvidence"}
