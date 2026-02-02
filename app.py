import streamlit as st
import pickle
from utils.text_preprocess import clean_text

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Disease Prediction App",
    page_icon="🩺",
    layout="centered"
)

# ---------------- LOAD MODEL ----------------
model = pickle.load(open("models/text_model.pkl", "rb"))
vectorizer = pickle.load(open("models/vectorizer.pkl", "rb"))

# ---------------- TITLE ----------------
st.title("🩺 Disease Prediction System")
st.write("Predict disease based on **symptoms entered by the user**")

# ---------------- INPUT ----------------
symptoms = st.text_area(
    "Enter your symptoms (comma separated)",
    placeholder="fever, headache, nausea"
)

# ---------------- PREDICTION ----------------
if st.button("Predict Disease"):
    if symptoms.strip() == "":
        st.error("❌ Please enter symptoms")
    else:
        clean_symptoms = clean_text(symptoms)
        vector = vectorizer.transform([clean_symptoms])

        prediction = model.predict(vector)[0]
        confidence = model.predict_proba(vector).max() * 100

        st.success(f"🧾 Predicted Disease: **{prediction}**")
        st.info(f"📊 Confidence: **{confidence:.2f}%**")

# ---------------- DISCLAIMER ----------------
st.warning("⚠️ This app is for educational purposes only. Consult a doctor.")
