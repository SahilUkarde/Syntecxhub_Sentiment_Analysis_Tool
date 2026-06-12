"""
predict.py
----------
Small command-line interface (CLI) for the Sentiment Analysis Tool.

Loads the trained model, TF-IDF vectorizer, and label encoder produced by
train.py, then predicts the sentiment of text supplied by the user.

Usage examples:
    # Interactive mode (keeps asking for input until you type 'exit' or 'quit')
    python src/predict.py

    # Single text passed directly as an argument
    python src/predict.py --text "I absolutely love this product!"

    # Show prediction probabilities/confidence as well
    python src/predict.py --text "This is terrible" --show-probs
"""

import argparse
import os
import sys

import joblib

from preprocess import clean_text

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "model")


def load_artifacts():
    model_path = os.path.join(MODEL_DIR, "sentiment_model.joblib")
    vectorizer_path = os.path.join(MODEL_DIR, "vectorizer.joblib")
    encoder_path = os.path.join(MODEL_DIR, "label_encoder.joblib")

    for p in (model_path, vectorizer_path, encoder_path):
        if not os.path.exists(p):
            sys.exit(
                f"Error: could not find '{p}'.\n"
                "Have you run 'python src/train.py' yet to train and save the model?"
            )

    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
    label_encoder = joblib.load(encoder_path)
    return model, vectorizer, label_encoder


def predict_sentiment(text, model, vectorizer, label_encoder, show_probs=False):
    cleaned = clean_text(text)
    if not cleaned:
        return "neutral", None  # nothing useful left after cleaning

    features = vectorizer.transform([cleaned])
    pred_idx = model.predict(features)[0]
    label = label_encoder.inverse_transform([pred_idx])[0]

    probs = None
    if show_probs and hasattr(model, "predict_proba"):
        proba = model.predict_proba(features)[0]
        probs = {cls: float(p) for cls, p in zip(label_encoder.classes_, proba)}

    return label, probs


def format_result(text, label, probs):
    emoji = {"positive": "🙂", "negative": "🙁", "neutral": "😐"}.get(label, "")
    lines = [f"Text     : {text}", f"Sentiment: {label.upper()} {emoji}"]
    if probs:
        lines.append("Confidence:")
        for cls, p in sorted(probs.items(), key=lambda x: -x[1]):
            lines.append(f"  {cls:<10}: {p:.2%}")
    return "\n".join(lines)


def interactive_loop(model, vectorizer, label_encoder, show_probs):
    print("=== Sentiment Analysis CLI ===")
    print("Type a sentence and press Enter to see its predicted sentiment.")
    print("Type 'exit' or 'quit' to stop.\n")
    while True:
        try:
            text = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if text.lower() in {"exit", "quit"}:
            print("Goodbye!")
            break
        if not text:
            continue

        label, probs = predict_sentiment(text, model, vectorizer, label_encoder, show_probs)
        print(format_result(text, label, probs))
        print()


def main():
    parser = argparse.ArgumentParser(description="Predict the sentiment of a piece of text.")
    parser.add_argument("--text", type=str, help="A single piece of text to classify.")
    parser.add_argument(
        "--show-probs",
        action="store_true",
        help="Show prediction confidence for each sentiment class.",
    )
    args = parser.parse_args()

    model, vectorizer, label_encoder = load_artifacts()

    if args.text:
        label, probs = predict_sentiment(
            args.text, model, vectorizer, label_encoder, args.show_probs
        )
        print(format_result(args.text, label, probs))
    else:
        interactive_loop(model, vectorizer, label_encoder, args.show_probs)


if __name__ == "__main__":
    main()
