import pickle

# Load saved model & vectorizer
with open("C:/Users/S rajiv gandhi/OneDrive/Desktop/Learning/Python/Sentimental_analysis/sentiment_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("C:/Users/S rajiv gandhi/OneDrive/Desktop/Learning/Python/Sentimental_analysis/vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

def predict_sentiment(text):
    """Predicts sentiment of the given review text."""
    processed_text = vectorizer.transform([text])  # Convert text to vector
    prediction = model.predict(processed_text)[0]  # Predict sentiment
    return "Positive 😃" if prediction == 1 else "Negative 😡"

if __name__ == "__main__":
    review = input("Enter a product review: ")
    print("Sentiment:", predict_sentiment(review))
