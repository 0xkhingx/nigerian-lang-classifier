#!/usr/bin/env python3
"""Quick demo of the Nigerian language classifier."""

from classifier import NigerianLangClassifier

clf = NigerianLangClassifier()

tests = [
    ("How are you doing today?", "English"),
    ("Bawo ni o se wa?", "Yoruba"),
    ("Kedu ka ị mere?", "Igbo"),
    ("Yaya dai?", "Hausa"),
    ("How you dey? Wetin dey happen?", "Pidgin"),
    ("The president gave a speech at the national assembly.", "English"),
    ("Mo nlo si ile iwe losi.", "Yoruba"),
    ("M na-aga ụlọ akwụkwọ.", "Igbo"),
    ("Ina zuwa makaranta.", "Hausa"),
    ("Dem go cook am for party.", "Pidgin"),
]
for text, expected in tests:
    result = clf.predict(text)
    proba = clf.predict_proba(text)
    top = max(proba, key=proba.get)
    ok = result == expected
    print(f"{'OK' if ok else 'FAIL':4} {text[:45]:<45} -> {result:<10} ({proba[result]:.0%})")
