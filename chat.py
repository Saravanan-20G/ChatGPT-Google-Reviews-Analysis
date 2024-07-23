import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import TruncatedSVD
import nltk
import re
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import emoji

# Load the model
model = joblib.load('best_model.pkl')

# Load the vectorizers and encoders
content_vectorizer = joblib.load('content_vectorizer.pkl')
token_content_vectorizer = joblib.load('token_content_vectorizer.pkl')
pos_tags_vectorizer = joblib.load('pos_tags_vectorizer.pkl')
review_created_version_encoder = joblib.load('review_created_version_encoder.pkl')
app_version_encoder = joblib.load('app_version_encoder.pkl')

# Download necessary NLTK data
nltk.download('stopwords')
nltk.download('punkt')

# Define text preprocessing functions
def preprocess_text(text):
    # Lowercase
    text = text.lower()
    
    # Remove HTML tags
    text = re.sub('<.*?>', '', text)
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Replace emojis with meanings
    text = emoji.demojize(text)
    
    # Expand chat words
    chat_words_mapping = {
        "lol": "laughing out loud",
        "brb": "be right back",
        "btw": "by the way",
        "afk": "away from keyboard",
        "rofl": "rolling on the floor laughing",
        "ttyl": "talk to you later",
        "np": "no problem",
        "thx": "thanks",
        "omg": "oh my god",
        "idk": "I don't know",
        "np": "no problem",
        "gg": "good game",
        "g2g": "got to go",
        "b4": "before",
        "cu": "see you",
        "yw": "you're welcome",
        "wtf": "what the f*ck",
        "imho": "in my humble opinion",
        "jk": "just kidding",
        "gf": "girlfriend",
        "bf": "boyfriend",
        "u": "you",
        "r": "are",
        "2": "to",
        "4": "for",
        "b": "be",
        "c": "see",
        "y": "why",
        "tho": "though",
        "smh": "shaking my head",
        "lolz": "laughing out loud",
        "h8": "hate",
        "luv": "love",
        "pls": "please",
        "sry": "sorry",
        "tbh": "to be honest",
        "omw": "on my way",
        "omw2syg": "on my way to see your girlfriend",
    }
    words = text.split()
    expanded_words = [chat_words_mapping.get(word.lower(), word) for word in words]
    text = ' '.join(expanded_words)
    
    # Remove stop words
    stop_words = set(stopwords.words('english'))
    tokens = word_tokenize(text)
    text = ' '.join([token for token in tokens if token not in stop_words])
    
    return text

def vectorize_text(text, vectorizer):
    return vectorizer.transform([text])

def encode_label(value, encoder):
    return encoder.transform([value])[0]

# Streamlit app
st.title("Review Score Prediction")

review_content = st.text_area("Enter the review content:")
review_created_version = st.text_input("Enter the review created version:")
app_version = st.text_input("Enter the app version:")
thumbs_up_count = st.number_input("Enter the thumbs up count:", min_value=0, step=1)

if st.button("Predict"):
    if review_content and review_created_version and app_version:
        # Preprocess the review content
        preprocessed_content = preprocess_text(review_content)
        
        # Vectorize the content
        vectorized_content = vectorize_text(preprocessed_content, content_vectorizer)
        
        # Encode the review created version and app version
        encoded_review_created_version = encode_label(review_created_version, review_created_version_encoder)
        encoded_app_version = encode_label(app_version, app_version_encoder)
        
        # Create the input data
        input_data = np.array([[vectorized_content.toarray()[0], encoded_review_created_version, encoded_app_version, thumbs_up_count]])
        
        # Make prediction
        prediction = model.predict(input_data)[0]
        
        st.write(f"Predicted Review Score: {prediction}")
    else:
        st.write("Please fill in all fields.")
