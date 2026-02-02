import streamlit as st
import pickle
from utils.text_preprocess import clean_text

st.set_page_config(page_title="Disease Predictor", page_icon="🩺")

# load model
model = pickle.load(open("models/text_model.pkl", "rb"))
vectorizer = pickle.load(open("models/vectorizer.pkl", "rb"))

st.title("🩺 Disease Prediction System")
st.write("Enter your symptoms to predict disease")

symptoms = st.text_area(
    "Symptoms (comma separated)",
    placeholder="fever, headache, nausea"
)

if st.button("Predict"):
    if symptoms.strip() == "":
        st.error("Please enter symptoms")
    else:
        clean = clean_text(symptoms)
        vect = vectorizer.transform([clean])
        result = model.predict(vect)[0]
        st.success(f"Predicted Disease: **{result}**")

st.warning("⚠️ This is for educational purposes only. Consult a doctor.")
