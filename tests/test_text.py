from legal_hse.text import lexical_tokenize


def test_legal_aware_tokenizer_preserves_refs_and_short_codes():
    tokens = lexical_tokenize(
        "ФИО подал иск по ст. 333.19 НК РФ и Федеральному закону 75-ФЗ по адрес.",
        min_len=2,
        lemmatize=True,
        preserve_legal_refs=True,
        legal_stop_words=True,
    )

    assert "фио" not in tokens
    assert "адрес" not in tokens
    assert "нк" in tokens
    assert "рф" in tokens
    assert "ст_333.19" in tokens
    assert any(token in tokens for token in ["75_фз", "75фз"])
    assert "федеральный" in tokens
