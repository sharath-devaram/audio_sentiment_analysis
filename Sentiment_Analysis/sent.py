import streamlit as st
import pandas as pd
import matplotlib.pyplot as mlt
import seaborn as se
import numpy as np
import plotly.express as px
import pickle
import click
import base64
import os
import numba
import librosa
import pyaudio
import wave
import joblib
import pickle
from audio_recorder_streamlit import audio_recorder
import librosa
import librosa.display
from keras.models import load_model
from sklearn.preprocessing import LabelEncoder
import plotly.graph_objects as go
from PIL import Image
image = Image.open('static/spade.png')

st.set_page_config(
    page_title="Audio Sentiment app by Millennial Garage",
    page_icon = image
)

image = Image.open('static/Capgemini_Logo.png')
image1 = st.sidebar.image("static/Capgemini_Logo.png", use_column_width=False, width=150)



# Path to folder containing audio files
audio_folder = r'C:\Users\DSHARATH\Downloads\Sentiment_Analysis\TESS Toronto emotional speech set data'

# List all files in the folder
audio_files = os.listdir(audio_folder)

# Iterate through files and load audio data
audio_data = {}
for file in audio_files:
    if file.endswith('.wav'):
        file_path = os.path.join(audio_folder, file)
        y, sr = librosa.load(file_path)
        audio_data[file] = {'y': y, 'sr': sr}

# Display audio players in Streamlit app
for file, data in audio_data.items():
    st.audio(data['y'], format='audio/wav', start_time=0, label=file)

st.title(':blue[Audio Sentiment Analysis]')
with st.sidebar:
    op = st.radio(" SELECT YOUR OPTION",["Audio Sentiment Analysis","Tone Modification"])
#if(op=="Data Loading"):
    #with open('model_SVM.pkl', 'rb') as f:
        #model = pickle.load(f)
if(op=="Audio Sentiment Analysis"):
    #if st.button("click here to upload the file"):
        # Define a function to extract audio features from an audio file
        model = joblib.load('Emotion_Voice_Detection_Model_SVM.pkl')
        def extract_features(audio_file):
    # Load the audio file using librosa
            audio, sr = librosa.load(audio_file, sr=None)

    # Extract the audio features using librosa
            features = np.array([
                np.mean(librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=20), axis=0),
                np.mean(librosa.feature.chroma_stft(y=audio, sr=sr), axis=0),
                np.mean(librosa.feature.spectral_contrast(y=audio, sr=sr), axis=0),
                np.mean(librosa.feature.tonnetz(y=audio, sr=sr), axis=0),
                np.mean(librosa.feature.rms(y=audio), axis=0),
                np.mean(librosa.feature.zero_crossing_rate(y=audio), axis=0),
                np.mean(librosa.effects.harmonic(y=audio), axis=0)
            ])
    # Return the features
            return features.reshape(1, -1)

# Define a function to predict the emotion of an audio file
        def predict_emotion(audio_file):
    # Extract the features from the audio file
            features = extract_features(audio_file)

    # Make a prediction using the model
            prediction = model.predict(features)[0]

    # Map the prediction to an emotion label
            emotion_labels = ['Angry', 'Happy', 'Neutral', 'Sad', 'Disgust', 'Pleasent_Suprise', 'Fear']
            predict_emotion = emotion_labels[np.argmax(prediction)]

    # Return the predicted emotion label
            return predicted_emotion

# Create a file uploader for WAV audio files
        uploaded_file = st.file_uploader("Choose a WAV audio file", type=["wav"])

# Add a "Predict Emotion" button to the Streamlit app
        if st.button("Predict Emotion") and uploaded_file is not None:
            predicted_emotion = predict_emotion(uploaded_file)

    # Show the predicted emotion label
            st.write("Predicted emotion:", predicted_emotion)
    
        
