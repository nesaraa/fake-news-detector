import streamlit as st
import joblib
from src.utils import clean_text

# Load model and vectorizer
model = joblib.load("models/fake_news_model.pkl")
vectorizer = joblib.load("models/vectorizer.pkl")

st.title("📰 Fake News Detector")
input_text = st.text_area("Enter the news article content:")

if st.button("Predict"):
    cleaned = clean_text(input_text)
    vec_text = vectorizer.transform([cleaned])
    result = model.predict(vec_text)[0]
    st.success("✅ Real News" if result == 1 else "🚨 Fake News")
