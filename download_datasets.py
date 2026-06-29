#!/usr/bin/env python3
"""Download public datasets for Nigerian languages.

Usage:
    python download_datasets.py           # print URLs and instructions
    python download_datasets.py --all     # attempt to download everything
"""

import argparse
import urllib.request
import zipfile
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data" / "raw"

URLS = {
    "naijasenti": "https://raw.githubusercontent.com/hausanlp/NaijaSenti/main/data/sentiment/en-twitter.zip",
    # "jw300": "https://object.pouta.csc.fi/OPUS-JW300/v1/tmx/en-yo.tmx.gz",
}

INSTRUCTIONS = """
────────────────────────────────────────────────────────────────
  Download public datasets to data/raw/
────────────────────────────────────────────────────────────────

  NaijaSenti (RECOMMENDED — real tweets, all 4 langs + Pidgin)
    https://github.com/hausanlp/NaijaSenti
    Download CSV files from data/sentiment/ directory
    Place in: data/raw/naijasenti/
      yoruba.csv  igbo.csv  hausa.csv  english.csv  pidgin.csv

  JW300 (clean parallel Bible text with full diacritics)
    https://opus.nlpl.eu/JW300.php
    Download en-yo.txt, en-ig.txt, en-ha.txt
    Place in: data/raw/jw300/
      en-yo.txt  en-ig.txt  en-ha.txt

  Masakhane News
    https://github.com/masakhane-io/masakhane-news
    Clone repo, copy language CSVs to data/raw/{lang}/

  Plain .txt files
    Any .txt in data/raw/{lang}/ with one sentence per line
    is auto-loaded on next train.

────────────────────────────────────────────────────────────────
  After downloading: python train.py --samples 1000
────────────────────────────────────────────────────────────────
"""


def main():
    parser = argparse.ArgumentParser(description="Download Nigerian language datasets")
    parser.add_argument("--all", action="store_true", help="Attempt to download all datasets")
    args = parser.parse_args()

    print(INSTRUCTIONS)

    if args.all:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        for name, url in URLS.items():
            dest = DATA_DIR / f"{name}.zip"
            print(f"Downloading {name} from {url}...")
            try:
                urllib.request.urlretrieve(url, dest)
                with zipfile.ZipFile(dest, "r") as z:
                    z.extractall(DATA_DIR / name)
                dest.unlink()
                print(f"  ✓ {name} extracted to data/raw/{name}/")
            except Exception as e:
                print(f"  ✗ {name} failed: {e}")


if __name__ == "__main__":
    main()
