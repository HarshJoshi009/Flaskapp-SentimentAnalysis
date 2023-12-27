from flask import Flask, render_template, request
import seaborn as sns
import matplotlib.pyplot as plt
from flask import Flask, render_template
from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import re
import string
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

app = Flask(__name__)


# Load the dataset
df = pd.read_csv("shuffled_dataset.csv", header=None,names=[ "text","target"])

df["text"] = df["text"].replace(np.nan, "", regex=True)

def normalize_text(text):
    if isinstance(text, str):
        text = text.lower()  # Convert to lowercase
        text = re.sub(r'\d+', '', text)  # Remove numbers (fixed regex)
        text = text.translate(str.maketrans('', '', string.punctuation))  # Remove punctuation
        text = ' '.join([word for word in text.split() if word not in ENGLISH_STOP_WORDS])  # Remove stop words
        text = text.strip()  # Remove leading/trailing spaces
    return text

df["text"] = df["text"].apply(normalize_text)

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(df["text"], df["target"], test_size=0.2, random_state=42)

# Create TF-IDF vectors
tfidf_vectorizer = TfidfVectorizer()
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

# Train an SVM classifier
svm_classifier = SVC(kernel='linear')
svm_classifier.fit(X_train_tfidf, y_train)


# Preprocess the user input text
def preprocess_text(text):
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'\d+', '', text)  # Remove numbers (fixed regex)
    text = text.translate(str.maketrans('', '', string.punctuation))  # Remove punctuation
    text = ' '.join([word for word in text.split() if word not in ENGLISH_STOP_WORDS])  # Remove stop words
    text = text.strip()  # Remove leading/trailing spaces
    return text

@app.route("/", methods=["GET", "POST"])
def index():
    sentiment = None
    if request.method == "POST":
        user_input = request.form["user_input"]
        preprocessed_input = preprocess_text(user_input)
        vector = tfidf_vectorizer.transform([preprocessed_input])
        sentiment = svm_classifier.predict(vector)[0]  
    return render_template("index.html", sentiment=sentiment)


if __name__ == '__main__':
    app.run(debug=True, port = 8000)

