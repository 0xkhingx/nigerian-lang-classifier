"""Train TF-IDF + Logistic Regression classifier with character n-grams."""

import re
import string
import pickle
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from .collector import collect_all

LANG_NAMES = {"en": "English", "yo": "Yoruba", "ig": "Igbo", "ha": "Hausa", "pcm": "Pidgin"}


def clean_text(text: str) -> str:
    """Basic text cleaning."""
    text = text.lower()
    text = re.sub(r"[{}]".format(re.escape(string.punctuation)), " ", text)
    text = re.sub(r"\d+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def prepare_data(target_per_lang: int = 200) -> tuple[list[str], list[str]]:
    """Collect and prepare training data."""
    raw = collect_all(target_per_lang)
    texts, labels = [], []
    for code, sentences in raw.items():
        for sent in sentences:
            texts.append(clean_text(sent))
            labels.append(LANG_NAMES[code])
    return texts, labels


def build_pipeline(
    use_char_ngrams: bool = True,
    max_features: int = 5000,
) -> Pipeline:
    """Build a TF-IDF + Logistic Regression pipeline.

    When use_char_ngrams is True, uses character n-grams (2-5),
    which typically outperforms word-level features on African
    languages due to rich morphology.
    """
    ngram_range = (2, 5) if use_char_ngrams else (1, 1)
    analyzer = "char" if use_char_ngrams else "word"

    vectorizer = TfidfVectorizer(
        analyzer=analyzer,
        ngram_range=ngram_range,
        max_features=max_features,
        sublinear_tf=True,
    )
    clf = LogisticRegression(max_iter=1000, C=1.0)
    return Pipeline([("tfidf", vectorizer), ("clf", clf)])


def train(
    target_per_lang: int = 200,
    use_char_ngrams: bool = True,
    model_path: str | Path = "model.pkl",
) -> Pipeline:
    """Run the full training pipeline."""
    print(f"Collecting {target_per_lang} sentences per language...")
    texts, labels = prepare_data(target_per_lang)

    print(f"Found {len(texts)} total sentences.")
    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels
    )

    pipe = build_pipeline(use_char_ngrams=use_char_ngrams)
    print("Training...")
    pipe.fit(X_train, y_train)

    acc = pipe.score(X_test, y_test)
    print(f"\nAccuracy: {acc:.2%}")
    print("\nClassification Report:")
    print(classification_report(y_test, pipe.predict(X_test)))

    with open(model_path, "wb") as f:
        pickle.dump(pipe, f)
    print(f"Model saved to {model_path}")

    return pipe
