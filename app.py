import streamlit as st
import pickle
from utils.text_preprocess import clean_text

# page config
st.set_page_config(
    page_title="Disease Prediction App",
    page_icon="🩺",
    layout="centered"
)

# load model
model = pickle.load(open("models/text_model.pkl", "rb"))
vectorizer = pickle.load(open("models/vectorizer.pkl", "rb"))

# title
st.title("🩺 Disease Prediction System")
st.write("Predict disease based on symptoms")

# input
symptoms = st.text_area(
    "Enter your symptoms (comma separated)",
    placeholder="fever, headache, nausea"
)

# predict button
if st.button("Predict Disease"):
    if symptoms.strip() == "":
        st.error("❌ Please enter symptoms")
    else:
        cleaned = clean_text(symptoms)
        vector = vectorizer.transform([cleaned])
        prediction = model.predict(vector)[0]

        st.success(f"🧾 Predicted Disease: **{prediction.upper()}**")

# disclaimer
st.warning("⚠️ This app is for educational purposes only. Consult a doctor.")
