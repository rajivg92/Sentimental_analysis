import pandas as pd
import re
import nltk
import pickle
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load dataset
df = pd.read_csv("C:/Users/S rajiv gandhi/OneDrive/Desktop/Learning/Python/Sentimental_analysis/Reviews.csv")

# Convert score to positive/negative sentiment
def convert_score_to_label(Score):
    return "negative" if Score <= 2 else "positive"

df["Sentiment"] = df["Score"].apply(convert_score_to_label)

# Preprocess text
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    text = ' '.join([word for word in text.split() if word not in stop_words])
    return text

df["Cleaned_Text"] = df["Text"].apply(clean_text)

# Convert labels to numbers (negative=0, positive=1)
df['Sentiment'] = df['Sentiment'].map({'negative': 0, 'positive': 1})

# Convert text into numerical vectors using TF-IDF
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df['Cleaned_Text'])
y = df['Sentiment']

# Split data into training & testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.6, random_state=42)

# Train a model
model = LogisticRegression()
model.fit(X_train, y_train)

# Evaluate model accuracy
y_pred = model.predict(X_test)
print("Model Accuracy:", accuracy_score(y_test, y_pred))

# Save model & vectorizer
with open("sentiment_model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

print("Model training complete. Saved as 'sentiment_model.pkl'")
