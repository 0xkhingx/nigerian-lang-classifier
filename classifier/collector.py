"""Collect sentences from Wikipedia for each language."""

import re
import urllib.request
import urllib.error
import json
from pathlib import Path

from data.sample_sentences import SENTENCES

WIKI_API = "https://{lang}.wikipedia.org/w/api.php"

LANG_CODES = {"en": "en", "yo": "yo", "ig": "ig", "ha": "ha"}

def fetch_wikipedia_sentences(lang: str, limit: int = 200) -> list[str]:
    """Fetch sentences from Wikipedia for a given language."""
    code = LANG_CODES.get(lang, lang)
    api = WIKI_API.format(lang=code)

    # Get random article titles
    params = (
        "?action=query"
        "&format=json"
        "&list=random"
        "&rnlimit=20"
        "&rnnamespace=0"
    )
    try:
        req = urllib.request.Request(api + params, headers={"User-Agent": "NigerianLangClassifier/1.0"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode())
    except Exception:
        return []

    titles = [p["title"] for p in data.get("query", {}).get("random", [])]
    sentences: list[str] = []

    for title in titles:
        if len(sentences) >= limit:
            break
        params = (
            "?action=query"
            "&format=json"
            "&titles=" + urllib.request.quote(title) +
            "&prop=extracts"
            "&explaintext=1"
            "&exsectionformat=plain"
            "&exlimit=1"
        )
        try:
            req = urllib.request.Request(api + params, headers={"User-Agent": "NigerianLangClassifier/1.0"})
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read().decode())
        except Exception:
            continue

        pages = data.get("query", {}).get("pages", {})
        for page in pages.values():
            text = page.get("extract", "")
            extracted = extract_sentences(text)
            sentences.extend(extracted)

    return sentences[:limit]


def extract_sentences(text: str) -> list[str]:
    """Split text into clean sentences."""
    text = re.sub(r"\s+", " ", text).strip()
    raw = re.split(r"(?<=[.!?])\s+", text)
    cleaned = []
    for s in raw:
        s = s.strip()
        if 10 < len(s) < 300:
            cleaned.append(s)
    return cleaned


def collect_all(target: int = 200) -> dict[str, list[str]]:
    """Collect sentences for all four languages. Falls back to samples."""
    data: dict[str, list[str]] = {}
    for lang in ["en", "yo", "ig", "ha"]:
        sentences = fetch_wikipedia_sentences(lang, target)
        if len(sentences) < target:
            fallback = SENTENCES.get(lang, [])
            needed = target - len(sentences)
            sentences.extend(fallback[:needed])
        data[lang] = sentences[:target]
    return data
