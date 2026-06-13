"""
app.py
------
Streamlit web app for the Sentiment Analysis Tool.

Run locally:
    pip install -r requirements.txt
    streamlit run app.py

Deploy for free:
    Push this repo to GitHub, then deploy on Streamlit Community Cloud
    (share.streamlit.io) — point it at app.py.
"""

import os

import joblib
import streamlit as st

from preprocess import clean_text

MODEL_DIR = os.path.dirname(os.path.abspath(__file__))


@st.cache_resource
def load_artifacts():
    model = joblib.load(os.path.join(MODEL_DIR, "sentiment_model.joblib"))
    vectorizer = joblib.load(os.path.join(MODEL_DIR, "vectorizer.joblib"))
    label_encoder = joblib.load(os.path.join(MODEL_DIR, "label_encoder.joblib"))
    return model, vectorizer, label_encoder


def predict_sentiment(text, model, vectorizer, label_encoder):
    cleaned = clean_text(text)
    if not cleaned:
        return "neutral", {}

    features = vectorizer.transform([cleaned])
    pred_idx = model.predict(features)[0]
    label = label_encoder.inverse_transform([pred_idx])[0]

    probs = {}
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(features)[0]
        probs = {cls: float(p) for cls, p in zip(label_encoder.classes_, proba)}

    return label, probs


# ---------------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Sentiment Analysis Tool",
    page_icon="💬",
    layout="centered",
)

st.title("💬 Sentiment Analysis Tool")
st.write(
    "Enter a tweet, review, or any short piece of text below and the model "
    "will predict whether the sentiment is **positive**, **negative**, or "
    "**neutral**."
)

model, vectorizer, label_encoder = load_artifacts()

# ---------------------------------------------------------------------------
# Input
# ---------------------------------------------------------------------------
text_input = st.text_area(
    "Your text",
    placeholder="e.g. The delivery was super fast and the staff were friendly!",
    height=120,
)

examples = {
    "👍 Positive example": "I absolutely love this product, it works perfectly!",
    "👎 Negative example": "This is the worst app I've ever used, totally broken.",
    "😐 Neutral example": "The package arrived on time, nothing special about it.",
}

st.write("Or try an example:")
cols = st.columns(len(examples))
for col, (label, example_text) in zip(cols, examples.items()):
    if col.button(label):
        text_input = example_text
        st.session_state["text_input"] = example_text

if "text_input" in st.session_state and not text_input:
    text_input = st.session_state["text_input"]

analyze = st.button("Analyze sentiment", type="primary")

# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------
if analyze or text_input:
    if not text_input.strip():
        st.warning("Please enter some text first.")
    else:
        label, probs = predict_sentiment(text_input, model, vectorizer, label_encoder)

        emoji = {"positive": "🙂", "negative": "🙁", "neutral": "😐"}.get(label, "")
        color = {"positive": "green", "negative": "red", "neutral": "gray"}.get(label, "gray")

        st.markdown(
            f"### Predicted sentiment: :{color}[{label.upper()} {emoji}]"
        )

        if probs:
            st.write("**Confidence scores:**")
            for cls, p in sorted(probs.items(), key=lambda x: -x[1]):
                st.write(f"{cls.capitalize()}")
                st.progress(p, text=f"{p:.1%}")

st.divider()
st.caption(
    "Model: TF-IDF + Logistic Regression, trained on labeled tweet/review data. "
    "See the project README for training and evaluation details."
)
