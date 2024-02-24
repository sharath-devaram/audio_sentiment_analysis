import streamlit as st
import pandas as pd
import matplotlib.pyplot as mlt
import seaborn as se
import numpy as np
import plotly.express as px
import pickle
import base64
import os
import librosa
import pyaudio
import wave
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

audio_folder = r'C:\Users\DIPMANI\Desktop\Sentiment_Analysis\TESS Toronto emotional speech set data'

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

    with open('Emotion_Voice_Detection_Model_SVM.pkl', 'rb') as f:

        model = pickle.load(f)

    emotion_labels = ['happy', 'sad', 'fear', 'disgust', 'pleasant surprise', 'angry', 'neutral']

    def extract_feature(file_name):

        try:

        # Load audio file

            audio, sr = librosa.load(file_name, res_type='kaiser_fast')


        # Extract features using Mel-Frequency Cepstral Coefficients (MFCCs)

            mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)


        # Resize to fixed shape

            mfccs = np.pad(mfccs, ((0, 0), (0, max(0, 100 - mfccs.shape[1]))), mode='constant')


        except Exception as e:

            print("Error encountered while parsing file: ", file_name)

            return None


        return mfccs

    def predict_emotion(file_name):

    # Extract audio features

        feature = extract_feature(file_name)


        if feature is None:

            return None

    # Flatten feature array for model prediction

        feature = feature.flatten()


    # Load label encoder

        label_encoder = LabelEncoder()

        label_encoder.classes_ = np.load('classes.py')

if(op=="Audio Sentiment Analysis"):

    def load_audio(audio_file_path):

        audio, sr = librosa.load(audio_file_path, res_type='kaiser_fast')

        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)

        mfccs_scaled = np.mean(mfccs.T, axis=0)

        return mfccs_scaled

    if st.button("Record Audio"):

        audio_bytes = audio_recorder()

        if audio_bytes:

            with open('audio.wav', 'wb') as f:

                f.write(audio_bytes)

            # Display recorded audio

        st.write('Recorded audio:')

        st.audio(audio_bytes, format='audio/wav')

    # Display audio waveform

        audio, sr = librosa.load('audio.wav', res_type='kaiser_fast')

        st.line_chart(audio)

        st.write("recorded audio", audio_bytes)

        mfccs = load_audio('audio.wav')

        mfccs = np.expand_dims(mfccs, axis=0)

        emotion_label = model.predict(mfccs)[0]

    if st.button("Predict Sentiment"):

        emotion_label = predict_emotion('audio.wav')

        st.write('Predicted emotion:', emotion_label)