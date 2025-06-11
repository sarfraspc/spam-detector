import streamlit as st
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import numpy as np
from scipy.sparse import hstack


model = joblib.load('../data/finalmodel.pkl')
vectorizer = joblib.load('../data/vectorizer.pkl')


nltk.download('stopwords')
nltk.download('wordnet')


stopword = set(stopwords.words('english'))
lemmetiser = WordNetLemmatizer()


def clean_text(text):
    text = text.lower()
    text = re.sub(r'\d+', ' ', text)
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    words = [lemmetiser.lemmatize(i) for i in text.split() if i not in stopword]
    return " ".join(words)  

st.title(' Spam Detector App')
st.markdown("Enter your message below to check if it's **Spam** or **Not Spam**.")

text = st.text_area(' Enter / paste a message')

if st.button('Predict'):
    if not text.strip():
        st.warning('Please enter a message!')
    else:
        cleaned = clean_text(text)
        vec = vectorizer.transform([cleaned])
        size = np.array([[len(text)]])
        final = hstack([vec, size])

        pred = model.predict(final)[0]
        pred_prob = model.predict_proba(final)[0]

        if pred == 1:
            st.error(f" Spam! (Confidence: {pred_prob[1]*100:.2f}%)")
        else:
            st.success(f" Not Spam! (Confidence: {pred_prob[0]*100:.2f}%)")
