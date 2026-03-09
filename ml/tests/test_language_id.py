from ml.pipeline.language_id import detect_language


def test_detect_language_basic():
    info_en = detect_language("This is an English sentence.")
    assert info_en.lang == "en"

    info_hi = detect_language("आज पेट्रोल बहुत महंगा हो गया है।")
    # langid may return 'hi' or similar; just check not English
    assert info_hi.lang != "en"