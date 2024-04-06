import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import string

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

# Step 1: Data Preprocessing
data = pd.read_csv("Stress.csv")

# Handle missing values if any
data.dropna(inplace=True)

# Tokenize the text
stop_words = set(stopwords.words('english'))
ps = PorterStemmer()

def preprocess_text(text):
    # Tokenize text
    tokens = word_tokenize(text)
    # Remove punctuation
    tokens = [word for word in tokens if word.isalnum()]
    # Convert to lowercase
    tokens = [word.lower() for word in tokens]
    # Remove stopwords
    tokens = [word for word in tokens if not word in stop_words]
    # Stemming
    tokens = [ps.stem(word) for word in tokens]
    return " ".join(tokens)

data['processed_text'] = data['text'].apply(preprocess_text)

# Step 2: Feature Extraction
tfidf_vectorizer = TfidfVectorizer(max_features=1000)  # You can adjust max_features as needed
X = tfidf_vectorizer.fit_transform(data['processed_text'])
y = data['label']

# Step 3: Model Training
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = MultinomialNB()
model.fit(X_train, y_train)

# Step 4: Model Evaluation
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))



# Example for making predictions
input_text = "Hey there r/assistance, Not sure if this is the right place to post this.. but here goes =) I'm currently a student intern at Sandia National Labs and working on a survey to help improve our marketing outreach efforts at the many schools we recruit at around the country. We're looking for current undergrad/grad STEM students so if you're a STEM student or know STEM students, I would greatly appreciate if you can help take or pass along this short survey. As a thank you, everyone who helps take the survey will be entered in to a drawing for chance to win one of three $50 Amazon gcs."

# Preprocess the input text
preprocessed_input = preprocess_text(input_text)

# Transform the preprocessed text using TF-IDF
input_vector = tfidf_vectorizer.transform([preprocessed_input])

# Make predictions
prediction = model.predict(input_vector)

print("Predicted class:", prediction)
