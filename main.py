# Step 1: Import Libraries and Load the Model
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model

# Load the IMDB dataset word index
word_index = imdb.get_word_index()
reverse_word_index = {value: key for key, value in word_index.items()}

# Load the pre-trained model with ReLU activation
model = load_model('simple_rnn_imdb.h5')

# Step 2: Helper Functions
# Function to decode reviews
def decode_review(encoded_review):
    return ' '.join([reverse_word_index.get(i - 3, '?') for i in encoded_review])

# Function to preprocess user input

# Function to preprocess user input
def preprocess_text(text):
    words = text.lower().split()

    encoded_review = [
        index + 3 if index is not None and index + 3 < 10000 else 2
        for word in words
        for index in [word_index.get(word)]
    ]

    padded_review = sequence.pad_sequences(
        [encoded_review],
        maxlen=500,
        padding='pre',
        truncating='pre'
    )

    return padded_review



import streamlit as st
## streamlit app
# Streamlit app
st.title('IMDB Movie Review Sentiment Analysis')
st.write('Enter a movie review to classify it as positive or negative.')

# User input
user_input = st.text_area('Movie Review')


if st.button('Classify'):

    if user_input.strip():

        preprocessed_input = preprocess_text(user_input)

        # Make prediction
        prediction = model.predict(preprocessed_input)
        score = float(prediction[0][0])

        sentiment = 'Positive' if score > 0.5 else 'Negative'

        # Display the result
        if sentiment == 'Positive':
            st.success(f'Sentiment: {sentiment}')
        else:
            st.error(f'Sentiment: {sentiment}')

        st.write(f'Prediction Score: {score:.4f}')

    else:
        st.warning('Please enter a movie review.')


