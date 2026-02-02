import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from utils.text_preprocess import clean_text

# load dataset
data = pd.read_csv("dataset/symptoms.csv")

# clean text
data["symptoms"] = data["symptoms"].apply(clean_text)

# vectorization
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(data["symptoms"])
y = data["disease"]

# train model
model = LogisticRegression()
model.fit(X, y)

# save model and vectorizer
pickle.dump(model, open("models/text_model.pkl", "wb"))
pickle.dump(vectorizer, open("models/vectorizer.pkl", "wb"))

print("✅ Text model trained and saved successfully")
