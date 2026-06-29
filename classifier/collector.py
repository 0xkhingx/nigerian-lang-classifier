"""Collect sentences from Wikipedia, public datasets, and embedded samples."""

import re
import csv
import urllib.request
import urllib.error
import json
from pathlib import Path

from data.sample_sentences import SENTENCES

WIKI_API = "https://{lang}.wikipedia.org/w/api.php"
LANG_CODES = {"en": "en", "yo": "yo", "ig": "ig", "ha": "ha"}
DATA_DIR = Path(__file__).parent.parent / "data"


def fetch_wikipedia_sentences(lang: str, limit: int = 200) -> list[str]:
    """Fetch sentences from Wikipedia for a given language."""
    code = LANG_CODES.get(lang, lang)
    api = WIKI_API.format(lang=code)
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


def load_jw300(lang: str, target: int = 200) -> list[str]:
    """Load from JW300 parallel corpus: data/raw/jw300/en-{lang}.txt"""
    path = DATA_DIR / "raw" / "jw300" / f"en-{lang}.txt"
    if not path.exists():
        return []
    sentences = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) == 2:
                sentences.append(parts[1])
    return extract_sentences(" ".join(sentences))[:target]


def load_naijasenti(lang: str, target: int = 200) -> list[str]:
    """Load from NaijaSenti data: data/raw/naijasenti/{lang}.tsv or .csv"""
    code = {"yo": "yor", "ig": "ibo", "ha": "hau", "en": "en", "pcm": "pcm"}.get(lang, lang)
    base = DATA_DIR / "raw" / "naijasenti"
    path = base / f"{code}.tsv"
    if not path.exists():
        path = base / f"{code}.csv"
    if not path.exists():
        return []
    sentences = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            tweet = row.get("tweet", row.get("text", "")).strip()
            if 10 < len(tweet) < 300:
                sentences.append(tweet)
    return sentences[:target]


def load_local_txt(lang: str, target: int = 200) -> list[str]:
    """Load from data/raw/{lang}/*.txt (one sentence per line)."""
    dir_path = DATA_DIR / "raw" / lang
    if not dir_path.exists():
        return []
    sentences = []
    for txt_file in sorted(dir_path.glob("*.txt")):
        with open(txt_file, "r", encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if 10 < len(s) < 300:
                    sentences.append(s)
    return sentences[:target]


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
    """Collect sentences for all four languages.

    Priority:
      1. NaijaSenti CSV in data/raw/naijasenti/
      2. JW300 parallel text in data/raw/jw300/
      3. Plain .txt files in data/raw/{lang}/
      4. Wikipedia live API
      5. Embedded sample sentences (fallback)
    """
    data: dict[str, list[str]] = {}
    for lang in ["en", "yo", "ig", "ha"]:
        sentences = load_naijasenti(lang, target)
        if len(sentences) < target:
            sentences.extend(load_jw300(lang, target - len(sentences)))
        if len(sentences) < target:
            sentences.extend(load_local_txt(lang, target - len(sentences)))
        if len(sentences) < target:
            sentences.extend(fetch_wikipedia_sentences(lang, target - len(sentences)))
        if len(sentences) < target:
            fallback = SENTENCES.get(lang, [])
            sentences.extend(fallback[: target - len(sentences)])
        data[lang] = sentences[:target]
    return data
