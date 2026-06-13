"""
train.py
--------
End-to-end training pipeline for the sentiment analysis tool.

Steps:
  1. Load labeled data from CSV (columns: "text", "label")
  2. Clean / preprocess the text
  3. Convert text to numeric features using TF-IDF
  4. Train two classifiers: Multinomial Naive Bayes and Logistic Regression
  5. Evaluate both on a held-out test set (accuracy, precision, recall, F1)
  6. Save the best-performing model + the fitted vectorizer + label encoder
     to the model/ directory using joblib, ready for the CLI to use.

Usage:
    python src/train.py --data data/sentiment_data.csv
"""

import argparse
import os

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder

from preprocess import clean_series

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "model")


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "text" not in df.columns or "label" not in df.columns:
        raise ValueError(
            "Input CSV must have columns named 'text' and 'label'. "
            f"Found columns: {list(df.columns)}"
        )
    df = df.dropna(subset=["text", "label"])
    return df


def evaluate(name, model, X_test, y_test, label_encoder):
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds, average="weighted")
    print(f"\n=== {name} ===")
    print(f"Accuracy: {acc:.4f}")
    print(f"Weighted F1: {f1:.4f}")
    print(
        classification_report(
            y_test, preds, target_names=label_encoder.classes_, zero_division=0
        )
    )
    return acc, f1


def main():
    parser = argparse.ArgumentParser(description="Train the sentiment analysis model.")
    parser.add_argument(
        "--data",
        default=os.path.join(os.path.dirname(__file__), "..", "data", "sentiment_data.csv"),
        help="Path to the labeled CSV dataset (columns: text, label).",
    )
    parser.add_argument(
        "--test-size", type=float, default=0.2, help="Fraction of data to use for testing."
    )
    args = parser.parse_args()

    print(f"Loading data from {args.data} ...")
    df = load_data(args.data)
    print(f"Loaded {len(df)} rows.")
    print("Label distribution:")
    print(df["label"].value_counts())

    print("\nCleaning text...")
    df["clean_text"] = clean_series(df["text"])

    # Encode string labels (positive/negative/neutral) to integers
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df["label"])

    X_train_text, X_test_text, y_train, y_test = train_test_split(
        df["clean_text"], y, test_size=args.test_size, random_state=42, stratify=y
    )

    print("\nVectorizing text with TF-IDF...")
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    X_train = vectorizer.fit_transform(X_train_text)
    X_test = vectorizer.transform(X_test_text)

    # ---- Train Multinomial Naive Bayes ----
    nb_model = MultinomialNB()
    nb_model.fit(X_train, y_train)
    nb_acc, nb_f1 = evaluate("Multinomial Naive Bayes", nb_model, X_test, y_test, label_encoder)

    # ---- Train Logistic Regression ----
    lr_model = LogisticRegression(max_iter=1000)
    lr_model.fit(X_train, y_train)
    lr_acc, lr_f1 = evaluate("Logistic Regression", lr_model, X_test, y_test, label_encoder)

    # ---- Pick the best model based on weighted F1 ----
    if lr_f1 >= nb_f1:
        best_model, best_name = lr_model, "Logistic Regression"
    else:
        best_model, best_name = nb_model, "Multinomial Naive Bayes"

    print(f"\nBest model: {best_name}")

    # ---- Save artifacts ----
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(best_model, os.path.join(MODEL_DIR, "sentiment_model.joblib"))
    joblib.dump(vectorizer, os.path.join(MODEL_DIR, "vectorizer.joblib"))
    joblib.dump(label_encoder, os.path.join(MODEL_DIR, "label_encoder.joblib"))

    with open(os.path.join(MODEL_DIR, "model_info.txt"), "w") as f:
        f.write(f"Best model: {best_name}\n")
        f.write(f"Accuracy: {max(nb_acc, lr_acc):.4f}\n")
        f.write(f"Weighted F1: {max(nb_f1, lr_f1):.4f}\n")
        f.write(f"Classes: {list(label_encoder.classes_)}\n")

    print(f"\nSaved model, vectorizer, and label encoder to '{MODEL_DIR}/'")


if __name__ == "__main__":
    main()
