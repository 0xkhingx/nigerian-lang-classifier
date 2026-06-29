"""Steps 2-6: comprehensive test of the Nigerian language classifier."""

import sys
import os
import pickle
import time
from pathlib import Path

# Simulate a fresh import from outside the project dir
sys.path.insert(0, r"C:\Users\HP\nigerian-lang-classifier")
os.chdir(os.path.dirname(os.path.abspath(__file__)))  # back to project dir for model.pkl

from classifier import NigerianLangClassifier

clf = NigerianLangClassifier()

SEP = "-" * 60

# ── Step 2: Basic sanity check ──────────────────────────────────
print(SEP)
print("STEP 2 — Basic sanity check")
print(SEP)
tests = [
    "How are you doing today?",
    "Bawo ni o se wa?",
    "Kedu ka i mere?",
    "Yaya dai?",
]
for t in tests:
    print(clf.predict(t), "-", t)

# ── Step 3: Edge cases ──────────────────────────────────────────
print()
print(SEP)
print("STEP 3 — Edge cases")
print(SEP)

# Short texts
print(clf.predict("Hello"), "- Hello")
print(clf.predict("e kaabo"), "- e kaabo (Yoruba)")

# Typos / mixed case
print(clf.predict("BAWO NI"), "- BAWO NI (Yoruba)")
print(clf.predict("kedu ka i mere"), "- kedu ka i mere (Igbo no diacritics)")

# Longer text
print(clf.predict(
    "Lagos is the most populous city in Nigeria and one of the largest in Africa."
), "- Long English")

# Tricky ones
print(clf.predict("Na so e be"), '- "Na so e be" (Pidgin)')
print(clf.predict("ok"), '- "ok" (too short)')
print(clf.predict("1234567"), '- "1234567" (numbers only)')

# ── Step 4: Confidence scores ────────────────────────────────────
print()
print(SEP)
print("STEP 4 — Confidence scores")
print(SEP)
high_confidence = [
    "The president addressed the nation yesterday.",
    "Mo nlo si ile iwe losi.",
    "M na-aga ulo akwukwo.",
    "Ina zuwa makaranta.",
]
for t in high_confidence:
    probs = clf.predict_proba(t)
    pred = clf.predict(t)
    conf = probs[pred]
    status = "OK" if conf > 0.8 else "LOW CONFIDENCE"
    print(f"{status:15} {conf:.0%}  {pred:10} {t[:40]}")

# ── Step 5: Real-world text ─────────────────────────────────────
print()
print(SEP)
print("STEP 5 — Real-world text (not from training data)")
print(SEP)
real_world = [
    ("E kaabo si ile wa", "Yoruba"),
    ("O bu ihe oma", "Igbo"),
    ("Allah ya kiyaye", "Hausa"),
    ("Nigeria go dey alright", "?"),  # Pidgin
]
for text, expected in real_world:
    pred = clf.predict(text)
    probs = clf.predict_proba(text)
    conf = probs[pred]
    match = "OK" if pred == expected else f"got {pred}"
    print(f"{match:15} {conf:.0%}  {text}")

# ── Step 6: Smoke test ──────────────────────────────────────────
print()
print(SEP)
print("STEP 6 — Smoke test (model size, load/inference time)")
print(SEP)
start = time.time()
with open("model.pkl", "rb") as f:
    pipe = pickle.load(f)
load_time = time.time() - start

start = time.time()
result = pipe.predict(["Bawo ni o se wa?"])
inference_time = time.time() - start

print(f"Model size: {Path('model.pkl').stat().st_size / 1024:.1f} KB")
print(f"Load time:  {load_time:.3f}s")
print(f"Inference:  {inference_time:.4f}s")
print(f"Result:     {result[0]}")
print(f"\nAll steps complete.")
