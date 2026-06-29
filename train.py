#!/usr/bin/env python3
"""Entry point: train the Nigerian language classifier."""

import argparse
from classifier.pipeline import train

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Nigerian language classifier")
    parser.add_argument("--samples", type=int, default=200, help="Sentences per language")
    parser.add_argument("--word-level", action="store_true", help="Use word-level features instead of char n-grams")
    parser.add_argument("--output", default="model.pkl", help="Output model path")
    args = parser.parse_args()

    train(
        target_per_lang=args.samples,
        use_char_ngrams=not args.word_level,
        model_path=args.output,
    )
