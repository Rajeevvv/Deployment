# -*- coding: utf-8 -*-
import pandas as pd
import streamlit as st
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
import re
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk 
nltk.download('stopwords')
nltk.download('punkt')

input_text = st.text_input("Enter text for Sentimental Analysis:")

if st.button("Clear"):
    input_text = ""

def clean_text(text):
    text = text.lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('<.*?>', '', text)
        
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
        
    # Tokenize text
    words = nltk.word_tokenize(text)
        
    # Remove stop words
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]
        
    # Lemmatize words
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]
        
    # Join words to form text
    text = ' '.join(words)
    return text

def rem(text):
  words=text.split()
  my_stopwords = stopwords.words('english')
  stopwords_to_add = ('mn','oyj','ab','inbev','ftsc','plc','afsc','eur','mln','hel','omx','esi')
  my_stopwords.extend(stopwords_to_add)
  filtered_words = [word for word in words if word.lower() not in my_stopwords]
  return ' '.join(filtered_words)


vectorizer = pickle.load(open('tf_idf_model.pkl','rb'))
model = pickle.load(open('mnb_model.plk','rb'))

if st.button('Analyze'):

    # 1. preprocess
    cleaned_text = clean_text(input_text)
    processed_text=rem(cleaned_text)
    
    # 2. vectorize
    vector_input = vectorizer.transform([processed_text])
    # 3. predict
    result = model.predict(vector_input)[0]
    # 4. Display
    if result == 0:
        st.header('Negative Statement')
    elif result == 1:
        st.header('Neutral statement')
    elif result == 2:
        st.header('Positive statement')

    

