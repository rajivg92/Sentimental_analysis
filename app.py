import streamlit as st
import pickle

# Load trained model & vectorizer
with open("C:/Users/S rajiv gandhi/OneDrive/Desktop/Learning/Python/Sentimental_analysis/sentiment_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("C:/Users/S rajiv gandhi/OneDrive/Desktop/Learning/Python/Sentimental_analysis/vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# Streamlit App UI
st.title("📢 Product Review Sentiment Analyzer")

review = st.text_area("Enter a product review:")

if st.button("Analyze Sentiment"):
    processed_review = vectorizer.transform([review])  # Convert text
    prediction = model.predict(processed_review)[0]  # Get prediction

    if prediction == 0:
        st.error("Negative 😡")
    else:
        st.success("Positive 😃")
