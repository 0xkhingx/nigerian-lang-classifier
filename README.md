# Nigerian Language Classifier

Detects **English**, **Yoruba**, **Igbo**, **Hausa**, and **Pidgin** from text.

**99.4% accuracy** on held-out test set (64k+ training sentences).

## Why it exists

Most NLP tools ignore Nigerian languages. This is a small fix.

## Quick start

```bash
pip install scikit-learn
python train.py
python demo.py
```

```python
from classifier import NigerianLangClassifier

clf = NigerianLangClassifier()
clf.predict("Bawo ni o se wa?")       # "Yoruba"
clf.predict("Kedu ka ị mere?")        # "Igbo"
clf.predict("Yaya dai?")              # "Hausa"
clf.predict("How you dey?")           # "Pidgin"
clf.predict("How is the weather?")    # "English"
```

## Datasets

The collector auto-loads whatever it can find, in priority order:

| Dataset | Languages | Type | Source |
|---------|-----------|------|--------|
| NaijaSenti | yo, ig, ha, pcm | Real tweets | [github.com/hausanlp/NaijaSenti](https://github.com/hausanlp/NaijaSenti) |
| JW300 | yo, ig, ha | Clean parallel text | [opus.nlpl.eu/JW300.php](https://opus.nlpl.eu/JW300.php) |
| Wikipedia API | en, yo, ig, ha | Random articles | live API |
| Gutenberg corpus | en | Books | NLTK |
| Embedded samples | en, yo, ig, ha | Hand-curated | ships with repo |

Place downloaded files in `data/raw/` — the collector picks them up automatically:

```
data/raw/
├── naijasenti/{yor,ibo,hau,pcm}/{train,dev,test}.tsv
├── jw300/               # TXT files: en-yo.txt, en-ig.txt, en-ha.txt
├── en/                  # Any .txt files, one sentence per line
├── yo/
├── ig/
└── ha/
```

Run `python download_datasets.py` for download URLs and instructions.

## Accuracy

99.4% on held-out test set (64k+ sentences across 5 languages).

### Real-world performance

| Input type | Works? |
|---|---|
| Clean English | ~95% confidence |
| Yoruba (no diacritics) | ~89% confidence |
| Yoruba (with diacritics) | ~97% confidence |
| Igbo | ~91% confidence |
| Hausa | ~97% confidence |
| Pidgin | ~82% confidence |
| Very short text (<3 words) | unreliable |

### Known limitations

- Pidgin has less training data than other languages
- Inputs under 5 words produce unreliable results
- Code-switching (mixed-language sentences) not handled
- English data relies on books and Wikipedia, not social media

## How it works

**Character n-grams (2–5) + TF-IDF + Logistic Regression.**

Char n-grams outperform word-level on morphologically rich languages like Yoruba and Igbo, where word boundaries don't carry as much signal.

```
text → char n-grams → TF-IDF vectorize → Logistic Regression → prediction
```

## What's next

- [ ] Expand Pidgin training data
- [ ] Expand diacritic-heavy training data for Yoruba/Igbo
- [ ] Wrap in FastAPI endpoint
- [ ] Deploy interactive demo to Hugging Face Spaces
