import gradio as gr
from classifier import NigerianLangClassifier

clf = NigerianLangClassifier()

LANG_FLAGS = {
    "English": "🇬🇧",
    "Yoruba": "🇳🇬",
    "Igbo": "🇳🇬",
    "Hausa": "🇳🇬",
    "Pidgin": "🇳🇬",
}

def classify(text):
    if not text or len(text.strip()) < 3:
        return "Too short to classify", {}
    pred = clf.predict(text)
    probs = clf.predict_proba(text)
    label = f"{LANG_FLAGS.get(pred, '')} {pred}"
    return label, probs

demo = gr.Interface(
    fn=classify,
    inputs=gr.Textbox(
        lines=3,
        placeholder="Paste text in English, Yoruba, Igbo, Hausa, or Pidgin...",
        label="Input Text"
    ),
    outputs=[
        gr.Text(label="Detected Language"),
        gr.Label(label="Confidence", num_top_classes=5)
    ],
    title="🇳🇬 Nigerian Language Classifier",
    description="Detects English, Yoruba, Igbo, Hausa, and Pidgin from text. "
                "Built with character n-grams + TF-IDF + Logistic Regression.",
    examples=[
        ["How are you doing today?"],
        ["Bawo ni o se wa?"],
        ["Kedu ka i mere?"],
        ["Ina zuwa makaranta."],
        ["How you dey? Wetin dey happen?"],
        ["Mo nlo si ile iwe losi."],
    ],
)

if __name__ == "__main__":
    demo.launch(theme=gr.themes.Soft())
