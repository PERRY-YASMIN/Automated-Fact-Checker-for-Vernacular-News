from ml.pipeline.retrieval import load_facts, build_fact_index, retrieve_for_claim


def test_retrieval_basic():
    facts = load_facts()
    emb, facts = build_fact_index(facts)
    claim = "500 और 1000 के नोट वापस आ रहे हैं"
    results = retrieve_for_claim(claim, emb, facts, top_k=3)
    assert len(results) >= 1