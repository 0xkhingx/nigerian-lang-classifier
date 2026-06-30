#!/usr/bin/env python3
"""Download public datasets for Nigerian languages.

Usage:
    python download_datasets.py           # print URLs and instructions
    python download_datasets.py --all     # attempt to download everything
"""

import argparse
import urllib.request
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data" / "raw"

NAIJASENTI_BASE = "https://raw.githubusercontent.com/hausanlp/NaijaSenti/main/data/annotated_tweets"
NAIJASENTI_LANGS = {
    "yor": ["train", "dev", "test"],
    "ibo": ["train", "dev", "test"],
    "hau": ["train", "dev", "test"],
    "pcm": ["train", "dev", "test"],
}

INSTRUCTIONS = """
-----------------------------------------------------------------
  Download public datasets to data/raw/
-----------------------------------------------------------------

  NaijaSenti (RECOMMENDED -- real tweets, all langs + Pidgin)
    python download_datasets.py --all
    Downloads train/dev/test TSVs for yor, ibo, hau, pcm

  JW300 (clean parallel Bible text with full diacritics)
    https://opus.nlpl.eu/JW300.php
    Download en-yo.txt, en-ig.txt, en-ha.txt
    Place in: data/raw/jw300/

  Masakhane News
    https://github.com/masakhane-io/masakhane-news
    Clone repo, copy language CSVs to data/raw/{lang}/

  Plain .txt files
    Any .txt in data/raw/{lang}/ with one sentence per line
    is auto-loaded on next train.

-----------------------------------------------------------------
  After downloading: python train.py --samples 15000
-----------------------------------------------------------------
"""


def download_naijasenti():
    """Download NaijaSenti annotated tweets for all available languages."""
    for code, splits in NAIJASENTI_LANGS.items():
        for split in splits:
            url = f"{NAIJASENTI_BASE}/{code}/{split}.tsv"
            dest_dir = DATA_DIR / "naijasenti" / code
            dest_dir.mkdir(parents=True, exist_ok=True)
            dest = dest_dir / f"{split}.tsv"
            print(f"Downloading {code}/{split}.tsv...")
            try:
                urllib.request.urlretrieve(url, dest)
                print(f"  -> {dest}")
            except Exception as e:
                print(f"  FAILED: {e}")


def main():
    parser = argparse.ArgumentParser(description="Download Nigerian language datasets")
    parser.add_argument("--all", action="store_true", help="Attempt to download all datasets")
    args = parser.parse_args()

    print(INSTRUCTIONS)

    if args.all:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        download_naijasenti()


if __name__ == "__main__":
    main()
