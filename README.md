# Syntecxhub_Sentiment_Analysis_Tool
Machine Learning-based Sentiment Analysis Tool using TF-IDF and Naive Bayes to classify text as Positive, Negative, or Neutral.

## 📌 Overview

This project demonstrates the implementation of a sentiment classification system using Python. The application preprocesses text data, converts it into numerical features using TF-IDF Vectorization, and predicts sentiment using a trained Machine Learning model.

The project was developed as part of an Artificial Intelligence Internship Program to gain hands-on experience in NLP and Machine Learning.

## 🚀 Features

* Text preprocessing and cleaning
* Sentiment classification (Positive, Negative, Neutral)
* TF-IDF feature extraction
* Machine Learning model training
* Interactive command-line interface (CLI)
* Model evaluation and prediction
* Pre-trained model storage using Joblib

## 🛠 Technologies Used

* Python
* Scikit-learn
* Pandas
* NumPy
* Joblib
* Natural Language Processing (NLP)

## 📂 Project Structure

```text
sentiment-analysis-tool/
├── data/
│   ├── generate_data.py   
│   └── sentiment_data.csv
│
├── model/
│   ├── label_encoder.joblib
│   ├── info.txt
│   ├── sentiment_model.joblib
│   └── vectorizer.joblib
│
├── src/
│   ├── predict.py
│   ├── preprocess.py
│   └── train.py
│
├── README.md
├── requirements.txt
└── .gitignore
```

## ⚙️ Installation

1. Clone the repository:

```bash
git clone https://github.com/your-username/Syntecxhub_Sentiment_Analysis_Tool.git
```

2. Navigate to the project folder:

```bash
cd Syntecxhub_Sentiment_Analysis_Tool
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

## ▶️ Usage

Train the model:

```bash
python src/train.py
```

Run sentiment prediction:

```bash
python src/predict.py
```

Enter any text and the model will predict whether the sentiment is Positive, Negative, or Neutral.

## 📊 Learning Outcomes

* Understanding NLP fundamentals
* Text preprocessing techniques
* Feature extraction using TF-IDF
* Machine Learning model training and evaluation
* Sentiment classification
* GitHub project management

## 🎯 Future Enhancements

* Web-based interface using Flask or Streamlit
* Deep Learning-based sentiment analysis
* Support for multiple languages
* Real-time social media sentiment tracking

## 👨‍💻 Author

**Sahil Ukarde**
B.E. Information Technology (2025)
Python Developer | Web Developer | AI/ML

## 📜 License

This project is developed for educational and internship purposes.
