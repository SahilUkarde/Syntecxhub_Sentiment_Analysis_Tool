# Syntecxhub Sentiment Analysis Tool

Machine Learning-based Sentiment Analysis Tool using TF-IDF and Logistic Regression / Naive Bayes to classify text as **Positive**, **Negative**, or **Neutral**.

## 🌐 Live Demo

**Try it here:** https://sahilukarde-syntecxhub-sentiment-analysis-tool-app-oezsnv.streamlit.app/

Enter any tweet, review, or short piece of text and instantly see the predicted sentiment along with confidence scores for each class.

## 📌 Overview

This project demonstrates an end-to-end sentiment classification pipeline built in Python. It:

1. Loads labeled text data (tweets / reviews).
2. Cleans and tokenizes the text (lowercasing, removing URLs, punctuation, stopwords, etc.).
3. Converts text into numeric features using **TF-IDF** vectorization.
4. Trains and evaluates **Naive Bayes** and **Logistic Regression** classifiers (accuracy & F1-score).
5. Saves the best-performing model for reuse.
6. Provides both a **command-line interface (CLI)** and a **web app** (Streamlit) for predicting sentiment on new text.

This project was developed as part of an Artificial Intelligence Internship Program to gain hands-on experience in NLP and Machine Learning.

## 🚀 Features

- Text preprocessing and cleaning
- Sentiment classification (Positive, Negative, Neutral)
- TF-IDF feature extraction
- Machine Learning model training and evaluation (accuracy, F1-score)
- Interactive command-line interface (CLI)
- Interactive web app with live demo (Streamlit)
- Pre-trained model storage using Joblib

## 🛠 Technologies Used

- Python
- Scikit-learn
- Pandas
- NumPy
- Joblib
- Streamlit
- Natural Language Processing (NLP)

## 📂 Project Structure

```text
Syntecxhub_Sentiment_Analysis_Tool/
├── app.py                  # Streamlit web app (live demo)
├── preprocess.py            # Text cleaning / preprocessing
├── sentiment_model.joblib   # Trained classifier
├── vectorizer.joblib         # Fitted TF-IDF vectorizer
├── label_encoder.joblib      # Label encoder (sentiment <-> class index)
├── info.txt                  # Model info (best model, accuracy, F1)
├── sentiment_pipeline_flow.png
├── Data/                      # Dataset and data generation script
│   ├── sentiment_data.csv
│   └── generate_data.py
├── src/                        # Training / prediction scripts
│   ├── train.py
│   └── predict.py
├── requirements.txt
├── README.md
├── LICENSE
└── .gitignore
```

> Note: `app.py` and `preprocess.py` must remain in the project root, alongside the three `.joblib` model files, for the live demo and CLI to load the model correctly.

## ⚙️ Setup

```bash
git clone https://github.com/SahilUkarde/Syntecxhub_Sentiment_Analysis_Tool.git
cd Syntecxhub_Sentiment_Analysis_Tool

python3 -m venv venv
source venv/bin/activate      # on Windows: venv\Scripts\activate

pip install -r requirements.txt
```

## ▶️ Usage

### Run the live web app locally

```bash
streamlit run app.py
```

This opens a browser tab with a text box, example buttons, and live sentiment predictions with confidence scores.

### Train the model

```bash
python src/train.py
```

This loads the dataset, cleans the text, extracts TF-IDF features, trains Naive Bayes and Logistic Regression, prints accuracy & F1-score for both, and saves the best model (`sentiment_model.joblib`, `vectorizer.joblib`, `label_encoder.joblib`) to the project root.

### Predict via the CLI

Single text:

```bash
python src/predict.py --text "I absolutely love this product!"
```

Show confidence scores:

```bash
python src/predict.py --text "This is the worst app I've ever used" --show-probs
```

Interactive mode:

```bash
python src/predict.py
```

## 📊 Model Performance

See `info.txt` for the latest training results, including the chosen model, accuracy, and weighted F1-score.

## 🧠 How It Works

1. **Preprocessing** (`preprocess.py`): lowercases text, strips URLs, @mentions, hashtags, punctuation, and numbers, then removes common English stopwords and very short tokens.
2. **Feature extraction**: a `TfidfVectorizer` (unigrams + bigrams) converts cleaned text into numeric vectors.
3. **Modeling**: `MultinomialNB` and `LogisticRegression` are trained on an 80/20 train/test split. The model with the higher weighted F1-score is saved.
4. **Inference**: both the CLI (`src/predict.py`) and the web app (`app.py`) reuse the exact same preprocessing and the saved vectorizer/model/label encoder to ensure consistent predictions.

## 👤 Author

Sahil Ukarde


BE IT Graduate | Python Developer | Web Developer | AI/ML

## 📄 License

This project is licensed under the terms of the [MIT License](LICENSE).
