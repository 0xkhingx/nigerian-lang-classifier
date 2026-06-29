"""Clean API for the Nigerian language classifier."""

import pickle
from pathlib import Path

from .pipeline import clean_text, build_pipeline, prepare_data

MODEL_PATH = Path(__file__).parent.parent / "model.pkl"


class NigerianLangClassifier:
    """Classify text as English, Yoruba, Igbo, or Hausa.

    Usage:
        clf = NigerianLangClassifier()
        clf.predict("How are you doing?")      # "English"
        clf.predict("Bawo ni o se wa?")       # "Yoruba"
        clf.predict("Kedu ka ị mere?")        # "Igbo"
        clf.predict("Yaya dai?")              # "Hausa"
    """

    def __init__(self, model_path: str | Path | None = None):
        path = Path(model_path) if model_path else MODEL_PATH
        if path.exists():
            with open(path, "rb") as f:
                self.pipeline_ = pickle.load(f)
        else:
            print("No trained model found. Training with defaults...")
            from .pipeline import train
            self.pipeline_ = train(model_path=path)

    def predict(self, text: str) -> str:
        """Predict the language of a single text string."""
        cleaned = clean_text(text)
        return self.pipeline_.predict([cleaned])[0]

    def predict_proba(self, text: str) -> dict[str, float]:
        """Return per-language probabilities for a text string."""
        cleaned = clean_text(text)
        probs = self.pipeline_.predict_proba([cleaned])[0]
        classes = self.pipeline_.classes_
        return dict(zip(classes, probs))

    def predict_batch(self, texts: list[str]) -> list[str]:
        """Predict language for a list of texts."""
        cleaned = [clean_text(t) for t in texts]
        return list(self.pipeline_.predict(cleaned))
