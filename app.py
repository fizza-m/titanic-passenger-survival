import os
import pickle

import numpy as np
import pandas as pd
import streamlit as st
from sklearn import svm
from sklearn.preprocessing import LabelEncoder

st.set_page_config(page_title="Titanic Survival Prediction", page_icon="🚢")

HERE = os.path.dirname(os.path.abspath(__file__))

# Same category lists as Step 5.1 of the Titanic notebook
CATEGORIES = {
    "PClass": ["First", "Second", "Third"],
    "Gender": ["Male", "Female"],
    "Sibling": ["Zero", "One", "Two", "Three"],
    "Embarked": ["Southampton", "Cherbourg", "Queenstown"],
}
FEATURES = list(CATEGORIES)


@st.cache_resource
def load_encoders():
    encoders = {}
    for name, values in CATEGORIES.items():
        enc = LabelEncoder()
        enc.fit(np.ravel(values))
        encoders[name] = enc
    return encoders


@st.cache_resource
def load_model(_encoders):
    """Load svc_trained_model.pkl from the notebook. If it cannot be loaded
    (e.g. scikit-learn version mismatch) and sample-data.csv is present,
    retrain the same SVC from it."""
    try:
        with open(os.path.join(HERE, "svc_trained_model.pkl"), "rb") as f:
            return pickle.load(f)
    except Exception:
        data = pd.read_csv(os.path.join(HERE, "sample-data.csv"))
        X = pd.DataFrame({c: _encoders[c].transform(data[c]) for c in FEATURES})
        y = survived_encoder().transform(data["Survived"])
        return svm.SVC(gamma="auto", random_state=0).fit(X, y)


def survived_encoder():
    enc = LabelEncoder()
    enc.fit(np.ravel(["Yes", "No"]))   # No = 0, Yes = 1
    return enc


encoders = load_encoders()
model = load_model(encoders)

st.title("Titanic Passenger Survival Prediction System")

pclass = st.selectbox("Passenger class (PClass)", CATEGORIES["PClass"])
gender = st.selectbox("Gender", CATEGORIES["Gender"])
sibling = st.selectbox("Siblings/spouses aboard (Sibling)", CATEGORIES["Sibling"])
embarked = st.selectbox("Port of embarkation (Embarked)", CATEGORIES["Embarked"])

if st.button("Predict"):
    row = pd.DataFrame({
        "PClass": [pclass], "Gender": [gender],
        "Sibling": [sibling], "Embarked": [embarked],
    })
    encoded = pd.DataFrame({c: encoders[c].transform(row[c]) for c in FEATURES})
    prediction = model.predict(encoded)[0]
    if prediction == 1:
        st.success("SURVIVED")
    else:
        st.error("NOT SURVIVED")
