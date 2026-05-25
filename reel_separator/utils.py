"""Utility functions: text normalization and filename sanitization."""

import re
import unicodedata


def normalize_text(text: str) -> str:
    """Normalize text for fuzzy comparison.

    Lowercase, strip accents, remove punctuation, collapse whitespace.
    """
    text = text.lower().strip()
    # Remove accents: "postproducción" -> "postproduccion"
    text = unicodedata.normalize("NFD", text)
    text = "".join(c for c in text if unicodedata.category(c) != "Mn")
    # Remove punctuation
    text = re.sub(r"[^\w\s]", "", text)
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


def sanitize_filename(title: str, index: int, extension: str = ".mp4") -> str:
    """Convert a reel title to a safe filename.

    Output: "01_IA_elimina_problema_green_screen.mp4"
    """
    name = re.sub(r'[<>:"/\\|?*]', "", title)
    name = name.strip().replace(" ", "_")
    # Truncate to reasonable length
    name = name[:80]
    return f"{index + 1:02d}_{name}{extension}"
