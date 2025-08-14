import streamlit as st
import pickle
import re
import nltk
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer

st.set_page_config(
    page_title="Thinkr",
    page_icon="📖"
)

nltk.download("stopwords")
nltk.download("wordnet")

stop_words = nltk.corpus.stopwords.words("english")

with open('models/ridge_alpha.pkl', 'rb') as file:
    ridge_alpha = pickle.load(file)

with open('models/vect.pkl', 'rb') as file:
    vect = pickle.load(file)

with open('models/logr.pkl', 'rb') as file:
    logr = pickle.load(file)

with open('models/logr_smote.pkl', 'rb') as file:
    logr_smote = pickle.load(file)

lemmatizer = WordNetLemmatizer()


def clean_text(txt):
    txt = re.sub('[^a-zA-Z]', ' ', txt)
    txt = txt.lower()
    txt = txt.split()
    txt = [lemmatizer.lemmatize(word) for word in txt if word not in stop_words]
    txt = ' '.join(txt)
    return txt


CountVec = CountVectorizer(ngram_range=(1,1), max_features=1000, stop_words='english')

st.title("Essay Evaluator")

st.write("Evaluate your essay and see how good your essay is. This model gives you a rough estimate of the quality of your essay. The"
         "model gives you a score from 1-6.")

essay = st.text_area("Enter your essay here : ")

source = st.text_area("Enter your source text here : ")

if st.button("Evaluate"):
    if essay and source:
        clean_essay = clean_text(essay)
        clean_source = clean_text(source)
        X = vect.transform([clean_essay, clean_source])
        pred = logr.predict(X)
        score = pred.mean()
        st.write(f"Your essay will score {score}")
    else:
        st.warning("Please fill in both the essay and the source text.")
