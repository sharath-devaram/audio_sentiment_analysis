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
##
import plotly.graph_objects as go
from audio_recorder_streamlit import audio_recorder
from playsound import playsound
import speech_recognition as sr 
from googletrans import Translator 
from gtts import gTTS 
import time
from deep_translator import GoogleTranslator
import streamlit as st
import warnings
warnings.filterwarnings('ignore')


audio_file = st.file_uploader("Choose a WAV audio file", type=["wav"])
time.sleep(10)
#st.write(audio_file.name)
#st.write(type(audio_file.name))

audio_file = open(audio_file.name, 'rb')
audio_bytes = audio_file.read()
st.success("Listen your translated audio here...")
st.audio(audio_bytes, format='audio/ogg')