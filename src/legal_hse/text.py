from __future__ import annotations

import re
from functools import lru_cache
from typing import Iterable

TOKEN_RE = re.compile(r"[а-яёa-z0-9]+", re.IGNORECASE)

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
) -> list[str]:
    tokens = [m.group(0).lower().replace("ё", "е") for m in TOKEN_RE.finditer(str(text))]
    if min_len > 1:
        tokens = [token for token in tokens if len(token) >= min_len]
    if stop_words:
        tokens = [token for token in tokens if token not in RU_STOP]
    if lemmatize:
        tokens = [lemmatize_token(token) for token in tokens]
    return tokens


def detokenize(tokens: Iterable[str]) -> str:
    return " ".join(tokens)


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
