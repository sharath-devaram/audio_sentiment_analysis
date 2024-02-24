#import methods 
import pandas as pd
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt
import librosa
import librosa.display
import IPython.display
import pyaudio
import soundfile as sf
import glob
import pickle
import wave
import scipy.signal as sig
import psola
from functools import partial
from pathlib import Path
from IPython.display import Audio
from IPython.display import Audio
from sklearn.model_selection import train_test_split
 # Import Decision Tree Classifier
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn import datasets
from sklearn.model_selection import KFold, cross_val_score
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
###xgb
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings('ignore')
import streamlit as st
from PIL import Image
#import Audio_Sentiment_Analysis
st.title("Tone Beautification")
######
##Image Background
import base64

def get_base64(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

def set_background(png_file):
    bin_str = get_base64(png_file)
    page_bg_img = '''
    <style>
    .stApp {
    background-image: url("data:image/png;base64,%s");
    background-size: cover;
    }
    </style>
    ''' % bin_str
    st.markdown(page_bg_img, unsafe_allow_html=True)

set_background('C:\\Users\\DSHARATH\\Downloads\\Sentiment_Analysis\\static\\Page33.png')
###
st.title("Tone Beautifier")
st.info("Application is running")
####
def correct(f0):
    if np.isnan(f0):
        return np.nan
    c_minor_degrees=librosa.key_to_degrees('C:min')
    c_minor_degrees=np.concatenate((c_minor_degrees,[c_minor_degrees[0]+12]))
    midi_note=librosa.hz_to_midi(f0)
    degree=midi_note%12
    closest_degree_id=np.argmin(np.abs(c_minor_degrees-degree))
    degree_difference=degree-c_minor_degrees[closest_degree_id]
    midi_note-=degree_difference
    return librosa.midi_to_hz(midi_note)
####
def correct_pitch(f0):
    corrected_f0=np.zeros_like(f0)
    for i in range(f0.shape[0]):
        corrected_f0[i]=correct(f0[i])
    smoothed_corrected_f0=sig.medfilt(corrected_f0, kernel_size=11)
    smoothed_corrected_f0[np.isnan(smoothed_corrected_f0)]=f0[np.isnan(smoothed_corrected_f0)]
    
    return smoothed_corrected_f0
####
def autotune(audio, sr):
    with st.spinner('Tone Beautifying is in progress...'):
        # Set some basis parameters.
        frame_length = 2048
        hop_length = frame_length // 4
        fmin = librosa.note_to_hz('C2')
        fmax = librosa.note_to_hz('C7')

        # Pitch tracking using the PYIN algorithm.
        f0, voiced_flag, voiced_probabilities = librosa.pyin(audio,
                                                         frame_length=frame_length,
                                                         hop_length=hop_length,
                                                         sr=sr,
                                                         fmin=fmin,
                                                         fmax=fmax)

    # Apply the chosen adjustment strategy to the pitch.
        corrected_f0 = correct_pitch(f0)

        # Pitch-shifting using the PSOLA algorithm.
    return psola.vocode(audio, sample_rate=int(sr), target_pitch=corrected_f0, fmin=fmin, fmax=fmax)
###

# Parse the command line arguments.
#option = st.selectbox("Select Option Here:",['File Upload'])
# print the selected hobby
#st.write("Selected option is : ",option)
#if (option=="File Upload"):
    #audio_file = st.file_uploader("Choose a WAV audio file", type=["wav"])

#filepath = Path(audio_file)
WAVE_OUTPUT_FILENAME_MODIFIED = "Modified_recording.wav"
# Parse the command line arguments.
filepath = Path('live_recording.wav')

# Load the audio file.
y, sr = librosa.load(filepath)

# Only mono-files are handled. If stereo files are supplied, only the first channel is used.
if y.ndim > 1:
    y = y[0, :]

# Perform the auto-tuning.
pitch_corrected_y = autotune(y, sr)

# Write the corrected audio to an output file.
filepath = filepath.parent / (filepath.stem + '_pitch_corrected' + filepath.suffix)
sf.write(str(WAVE_OUTPUT_FILENAME_MODIFIED), pitch_corrected_y, sr)
audio_files = glob.glob('*.wav') # Replace '*.wav' with the file extension of your audio files
#print(audio_files)
for file in audio_files:
    if file=="modified_recording.wav":
        fileName=os.path.basename(file)
        beauty_modified_audio = Audio(file)
    if file=="live_recording.wav":
        fileName_original=os.path.basename(file)
        beauty_original_audio=Audio(file)
        
    
audio_file = open('modified_recording.wav', 'rb')
audio_bytes = audio_file.read()
st.info("Beautified Audio")
st.audio(audio_bytes, format='audio/ogg')

sample_rate = 44100  # 44100 samples per second
seconds = 2  # Note duration of 2 seconds
frequency_la = 440  # Our played note will be 440 Hz
# Generate array with seconds*sample_rate steps, ranging between 0 and seconds
t = np.linspace(0, seconds, seconds * sample_rate, False)
# Generate a 440 Hz sine wave
note_la = np.sin(frequency_la * t * 2 * np.pi)

#st.audio(note_la, sample_rate=sample_rate)
audio_file = open('live_recording.wav', 'rb')
audio_bytes = audio_file.read()
st.info("Original Audio")
st.audio(audio_bytes, format='audio/ogg')

sample_rate = 44100  # 44100 samples per second
seconds = 2  # Note duration of 2 seconds
frequency_la = 440  # Our played note will be 440 Hz
# Generate array with seconds*sample_rate steps, ranging between 0 and seconds
t = np.linspace(0, seconds, seconds * sample_rate, False)
# Generate a 440 Hz sine wave
note_la = np.sin(frequency_la * t * 2 * np.pi)
st.success("Done...!")
st.success("Thank You 🙂")