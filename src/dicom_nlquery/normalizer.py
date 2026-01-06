from __future__ import annotations

import unicodedata


def normalize(text: str) -> str:
    if not text:
        return ""
    decomposed = unicodedata.normalize("NFD", text)
    ascii_text = decomposed.encode("ASCII", "ignore").decode("ASCII")
    return " ".join(ascii_text.lower().split())
