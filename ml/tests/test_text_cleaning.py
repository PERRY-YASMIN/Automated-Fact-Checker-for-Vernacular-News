from ml.pipeline.text_cleaning import normalize_for_embedding, simple_sentence_split


def test_normalize_and_split():
    text = "Good morning! Petrol is 120 rupees now!!! Please share. https://example.com"
    norm = normalize_for_embedding(text)
    assert "http" not in norm
    assert "please share".lower() not in norm.lower()

    sentences = simple_sentence_split(norm)
    assert len(sentences) >= 1