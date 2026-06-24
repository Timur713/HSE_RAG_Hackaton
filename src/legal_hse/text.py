from __future__ import annotations

import re
from functools import lru_cache
from typing import Iterable

TOKEN_RE = re.compile(r"[а-яёa-z0-9]+", re.IGNORECASE)
LEGAL_REF_RE = re.compile(
    r"""
    (?:
        \bст\.?\s*\d+(?:[.\-]\d+)*\b
        |
        \bп\.?\s*\d+(?:[.\-]\d+)*\b
        |
        \bч\.?\s*\d+(?:[.\-]\d+)*\b
        |
        \b\d+(?:[.\-]\d+)*\s*[- ]?\s*фз\b
        |
        \b\d+(?:[.\-]\d+)+\b
    )
    """,
    re.IGNORECASE | re.VERBOSE,
)

LEGAL_SHORT_TOKENS = {
    "гк",
    "гпк",
    "кас",
    "нк",
    "опс",
    "пфр",
    "рф",
    "сз",
    "тк",
    "ук",
    "фз",
    "цб",
}

LEGAL_NOISE_STOP = {
    "адрес",
    "дата",
    "тел",
    "телефон",
    "фио",
}

RU_STOP = {
    "а",
    "без",
    "бы",
    "был",
    "была",
    "были",
    "было",
    "в",
    "вам",
    "вас",
    "весь",
    "во",
    "вот",
    "все",
    "всего",
    "вы",
    "где",
    "да",
    "для",
    "до",
    "его",
    "ее",
    "если",
    "есть",
    "же",
    "за",
    "и",
    "из",
    "или",
    "им",
    "их",
    "к",
    "как",
    "ко",
    "ли",
    "мне",
    "мы",
    "на",
    "над",
    "не",
    "него",
    "нее",
    "нет",
    "ни",
    "но",
    "о",
    "об",
    "он",
    "она",
    "они",
    "оно",
    "от",
    "по",
    "под",
    "при",
    "с",
    "со",
    "так",
    "то",
    "у",
    "уже",
    "чем",
    "что",
    "чтобы",
    "это",
    "я",
}


def normalize_text(text: str, *, replace_yo: bool = True) -> str:
    text = "" if text is None else str(text)
    text = text.replace("\u00a0", " ")
    text = text.replace("«", '"').replace("»", '"')
    text = text.replace("–", "-").replace("—", "-")
    if replace_yo:
        text = text.replace("ё", "е").replace("Ё", "Е")
    return re.sub(r"\s+", " ", text).strip().lower()


def lexical_tokenize(
    text: str,
    *,
    min_len: int = 3,
    stop_words: bool = True,
    lemmatize: bool = False,
    preserve_legal_refs: bool = False,
    legal_stop_words: bool = False,
    add_bigrams: bool = False,
) -> list[str]:
    normalized = normalize_text(str(text))
    tokens = [m.group(0).lower().replace("ё", "е") for m in TOKEN_RE.finditer(normalized)]
    if preserve_legal_refs:
        tokens.extend(_legal_ref_tokens(normalized))
    if min_len > 1:
        tokens = [
            token
            for token in tokens
            if len(token) >= min_len or token in LEGAL_SHORT_TOKENS or _is_structured_token(token)
        ]
    if stop_words:
        tokens = [token for token in tokens if token not in RU_STOP]
    if lemmatize:
        tokens = [_safe_lemmatize_token(token) for token in tokens]
    if legal_stop_words:
        tokens = [token for token in tokens if token not in LEGAL_NOISE_STOP]
    if add_bigrams:
        tokens = tokens + _adjacent_bigrams(tokens)
    return tokens


def detokenize(tokens: Iterable[str]) -> str:
    return " ".join(tokens)


def _legal_ref_tokens(text: str) -> list[str]:
    tokens: list[str] = []
    for match in LEGAL_REF_RE.finditer(text):
        token = re.sub(r"\s+", "", match.group(0).lower().replace("ё", "е"))
        token = token.replace("ст.", "ст_").replace("п.", "п_").replace("ч.", "ч_")
        token = token.replace("ст", "ст_", 1) if token.startswith("ст") and not token.startswith("ст_") else token
        token = token.replace("п", "п_", 1) if token.startswith("п") and not token.startswith("п_") else token
        token = token.replace("ч", "ч_", 1) if token.startswith("ч") and not token.startswith("ч_") else token
        token = token.replace("-фз", "_фз")
        if token:
            tokens.append(token)
    return tokens


def _is_structured_token(token: str) -> bool:
    return any(char.isdigit() for char in token) or "_" in token or "." in token or "-" in token


def _safe_lemmatize_token(token: str) -> str:
    if token in LEGAL_SHORT_TOKENS or _is_structured_token(token):
        return token
    return lemmatize_token(token)


def _adjacent_bigrams(tokens: list[str]) -> list[str]:
    bigrams: list[str] = []
    for left, right in zip(tokens, tokens[1:], strict=False):
        if _is_structured_token(left) or _is_structured_token(right):
            continue
        if left in RU_STOP or right in RU_STOP:
            continue
        if left in LEGAL_NOISE_STOP or right in LEGAL_NOISE_STOP:
            continue
        bigrams.append(f"{left}__{right}")
    return bigrams


@lru_cache(maxsize=100_000)
def lemmatize_token(token: str) -> str:
    """Lemmatize if pymorphy3 is installed; otherwise keep the token unchanged."""

    try:
        import pymorphy3  # type: ignore
    except ImportError:
        return token

    morph = _get_morph()
    return morph.parse(token)[0].normal_form


@lru_cache(maxsize=1)
def _get_morph():
    import pymorphy3  # type: ignore

    return pymorphy3.MorphAnalyzer()
