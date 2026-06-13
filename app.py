"""
app_gradio.py
-------------
Gradio web app for the Sentiment Analysis Tool (alternative to app.py/Streamlit).

Run locally:
    pip install -r requirements.txt
    python app_gradio.py

Deploy for free on Hugging Face Spaces:
    1. Create a new Space (SDK: Gradio).
    2. Upload this repo's files. Make sure this file is named `app.py`
       in the Space (Hugging Face looks for app.py by default), or set
       the Space's "app file" setting to app_gradio.py.
    3. The Space will install requirements.txt and launch automatically,
       giving you a public URL.
"""

import os

import gradio as gr
import joblib

from preprocess import clean_text

MODEL_DIR = os.path.dirname(os.path.abspath(__file__))

model = joblib.load(os.path.join(MODEL_DIR, "sentiment_model.joblib"))
vectorizer = joblib.load(os.path.join(MODEL_DIR, "vectorizer.joblib"))
label_encoder = joblib.load(os.path.join(MODEL_DIR, "label_encoder.joblib"))

EMOJI = {"positive": "🙂", "negative": "🙁", "neutral": "😐"}


def predict_sentiment(text):
    cleaned = clean_text(text)
    if not cleaned:
        return "Please enter some text.", {}

    features = vectorizer.transform([cleaned])
    pred_idx = model.predict(features)[0]
    label = label_encoder.inverse_transform([pred_idx])[0]

    probs = {}
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(features)[0]
        probs = {cls: float(p) for cls, p in zip(label_encoder.classes_, proba)}

    result = f"{label.upper()} {EMOJI.get(label, '')}"
    return result, probs


examples = [
    "I absolutely love this product, it works perfectly!",
    "This is the worst app I've ever used, totally broken.",
    "The package arrived on time, nothing special about it.",
]

demo = gr.Interface(
    fn=predict_sentiment,
    inputs=gr.Textbox(
        lines=3,
        placeholder="e.g. The delivery was super fast and the staff were friendly!",
        label="Your text",
    ),
    outputs=[
        gr.Textbox(label="Predicted sentiment"),
        gr.Label(label="Confidence scores", num_top_classes=3),
    ],
    examples=examples,
    title="💬 Sentiment Analysis Tool",
    description=(
        "Enter a tweet, review, or any short piece of text and the model will "
        "predict whether the sentiment is **positive**, **negative**, or **neutral**. "
        "Model: TF-IDF + Logistic Regression."
    ),
)

if __name__ == "__main__":
    demo.launch()
