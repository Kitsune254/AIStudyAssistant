import streamlit as st
import pickle
import re
import nltk
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer

nltk.download("stopwords")
nltk.download("wordnet")

stop_words = nltk.corpus.stopwords.words("english")

with open('models/ridge_alpha.pkl', 'rb') as file:
    ridge_alpha = pickle.load(file)

lemmatizer = WordNetLemmatizer()

def clean_text(txt):
    txt = re.sub('[^a-zA-Z]', ' ', txt)
    txt = txt.lower()
    txt = txt.split()
    txt = [lemmatizer.lemmatize(word) for word in txt if not word in stop_words]
    txt = ' '.join(txt)
    return txt

Count_vec = CountVectorizer(ngram_range=(1,1), max_features=1000, stop_words='english')
Count_data = Count_vec.fit_transform()


