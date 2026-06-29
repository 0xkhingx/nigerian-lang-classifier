# Nigerian Language Classifier

Detects **English**, **Yoruba**, **Igbo**, and **Hausa** from text.

97.8% accuracy across 4 languages.

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

## Results

| Language   | Precision | Recall | F1-score |
|-----------|-----------|--------|----------|
| English   | 100%      | 100%   | 1.00     |
| Yoruba    | 94%       | 97%    | 0.95     |
| Igbo      | 100%      | 97%    | 0.98     |
| Hausa     | 97%       | 97%    | 0.97     |

## How it works

**Character n-grams (2–5) + TF-IDF + Logistic Regression.**

Char n-grams outperform word-level on morphologically rich languages like Yoruba and Igbo, where word boundaries don't carry as much signal.

```
text → char n-grams → TF-IDF vectorize → Logistic Regression → prediction
```

## Extend it

- Add Pidgin English
- Add Efik, Tiv, Ijaw
- Fine-tune on domain-specific text
- Swap in an MLP or FastText for higher accuracy

**Push it to GitHub today. That's step one.**
