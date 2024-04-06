from flask import Flask, render_template, request, jsonify
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import string

app = Flask(__name__)

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

# Load the trained model
model = MultinomialNB()
tfidf_vectorizer = TfidfVectorizer(max_features=1000)  # Same vectorizer used during training

data = pd.read_csv("Stress.csv")
data.dropna(inplace=True)

stop_words = set(stopwords.words('english'))
ps = PorterStemmer()

def preprocess_text(text):
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word.isalnum()]
    tokens = [word.lower() for word in tokens]
    tokens = [word for word in tokens if not word in stop_words]
    tokens = [ps.stem(word) for word in tokens]
    return " ".join(tokens)

data['processed_text'] = data['text'].apply(preprocess_text)

X = tfidf_vectorizer.fit_transform(data['processed_text'])
y = data['label']
model.fit(X, y)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    input_text = request.form['text']
    preprocessed_input = preprocess_text(input_text)
    input_vector = tfidf_vectorizer.transform([preprocessed_input])
    prediction = int(model.predict(input_vector)[0])  # Convert prediction to Python integer
    return jsonify({'prediction': prediction})

if __name__ == '__main__':
    app.run(debug=True)
