import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from utils.text_preprocess import clean_text

# load kaggle dataset
data = pd.read_csv("dataset/Training.csv")

# separate symptoms and disease
X = data.drop("prognosis", axis=1)
y = data["prognosis"]

# convert 0/1 columns → text
symptom_names = X.columns

def row_to_text(row):
    return " ".join(symptom_names[row == 1])

symptom_text = X.apply(row_to_text, axis=1)
symptom_text = symptom_text.apply(clean_text)

# vectorization
vectorizer = TfidfVectorizer()
X_vector = vectorizer.fit_transform(symptom_text)

# model
model = LogisticRegression(max_iter=1000)
model.fit(X_vector, y)

# save model
pickle.dump(model, open("models/text_model.pkl", "wb"))
pickle.dump(vectorizer, open("models/vectorizer.pkl", "wb"))

print("✅ Model trained using Kaggle dataset")
