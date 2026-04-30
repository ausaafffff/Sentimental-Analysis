import streamlit as st
import pickle
import re

model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

def clean_text(text):
    text = str(text)
    text = text.lower()
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    return text

def predict(text):
    cleaned = clean_text(text)
    vec = vectorizer.transform([cleaned]).toarray()
    result = model.predict(vec)[0]   
    return "Positive " if result == 1 else "Negative "

st.title(" Sentiment Analysis App")
st.write("Enter any review or sentence below:")

user_input = st.text_area("Your Text Here", height=150)

if st.button("Predict Sentiment"):
    if user_input.strip() == "":
        st.warning("Please enter some text first!")
    else:
        result = predict(user_input)
        st.success(f"Sentiment: **{result}**")
        