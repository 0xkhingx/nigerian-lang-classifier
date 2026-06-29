# Nigerian Language Classifier

Detects **English**, **Yoruba**, **Igbo**, and **Hausa** from text.

97.8% accuracy on clean held-out test sentences.

## Why it exists

Most NLP tools ignore Nigerian languages. This is a small fix.

## Quick start

```bash
pip install scikit-learn nltk
python train.py
python demo.py
```

```python
from classifier import NigerianLangClassifier

clf = NigerianLangClassifier()
clf.predict("Bawo ni o se wa?")       # "Yoruba"
clf.predict("Kedu ka ị mere?")        # "Igbo"
clf.predict("Yaya dai?")              # "Hausa"
clf.predict("How is the weather?")    # "English"
```

## Datasets

The collector auto-loads whatever it can find, in priority order:

| Dataset | Languages | Type | Source |
|---------|-----------|------|--------|
| NaijaSenti | yo, ig, ha, en, pcm | Real tweets | [github.com/hausanlp/NaijaSenti](https://github.com/hausanlp/NaijaSenti) |
| JW300 | yo, ig, ha | Clean parallel text | [opus.nlpl.eu/JW300.php](https://opus.nlpl.eu/JW300.php) |
| Wikipedia API | yo, ig, ha, en | Random articles | live API |
| Embedded samples | yo, ig, ha, en | Hand-curated | ships with repo |

Place downloaded files in `data/raw/` — the collector picks them up automatically:

```
data/raw/
├── naijasenti/          # CSV files: yoruba.csv, igbo.csv, hausa.csv, english.csv, pidgin.csv
├── jw300/               # TXT files: en-yo.txt, en-ig.txt, en-ha.txt
├── yo/                  # Any .txt files, one sentence per line
├── ig/
└── ha/
```

Run `python download_datasets.py` for download URLs and instructions.

## Accuracy

98% on clean, well-formed sentences (held-out test set).

### Real-world performance

| Input type | Works? |
|---|---|
| Clean English | ~86% confidence |
| Yoruba (no diacritics) | ~84% confidence |
| Yoruba (with diacritics) | improving |
| Igbo (no diacritics) | ~74% confidence |
| Hausa | ~63% confidence |
| Pidgin English | not supported yet |
| Very short text (<3 words) | unreliable |

### Known limitations

- Real Yoruba text with full tonal marks (ẹ, ọ, ṣ) needs more training data
- Pidgin English is not yet a supported class
- Inputs under 5 words produce unreliable results
- Code-switching (mixed-language sentences) not handled

## How it works

**Character n-grams (2–5) + TF-IDF + Logistic Regression.**

Char n-grams outperform word-level on morphologically rich languages like Yoruba and Igbo, where word boundaries don't carry as much signal.

```
text → char n-grams → TF-IDF vectorize → Logistic Regression → prediction
```

## What's next

- [ ] Add Pidgin English as a fifth class
- [ ] Expand diacritic-heavy training data for Yoruba/Igbo
- [ ] Wrap in FastAPI endpoint
- [ ] Deploy interactive demo to Hugging Face Spaces
