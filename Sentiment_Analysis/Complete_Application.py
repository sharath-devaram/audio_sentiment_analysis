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
from PIL import Image
image = Image.open('static/spade.png')

st.set_page_config(
    page_title="Audio Sentiment app by Millennial Garage",
    page_icon = image)

image = Image.open('static/Capgemini_Logo.png')
image1 = st.sidebar.image("static/Capgemini_Logo.png", use_column_width=False, width=150)
# Define the Streamlit app
st.title("Audio Sentiment Analysis")
####
###Translation
##Language Types
dic=('afrikaans', 'af', 'albanian', 'sq', 'amharic', 'am', 'arabic', 'ar', 'armenian', 'hy', 'azerbaijani', 'az',
 'basque', 'eu', 'belarusian', 'be', 'bengali', 'bn', 'bosnian','bs', 'bulgarian', 'bg', 'catalan', 'ca',
  'cebuano', 'ceb', 'chichewa', 'ny', 'chinese (simplified)', 'zh-cn', 'chinese (traditional)', 'zh-tw', 'corsican', 'co', 'croatian', 'hr', 'czech', 'cs', 'danish',
     'da', 'dutch', 'nl', 'english', 'en', 'esperanto',
  'eo', 'estonian', 'et', 'filipino', 'tl', 'finnish', 'fi', 
     'french', 'fr', 'frisian', 'fy', 'galician', 'gl',
  'georgian', 'ka', 'german', 'de', 'greek', 'el', 'gujarati', 
     'gu', 'haitian creole', 'ht', 'hausa', 'ha', 
  'hawaiian', 'haw', 'hebrew', 'he', 'hindi', 'hi', 'hmong', 
     'hmn', 'hungarian', 'hu', 'icelandic', 'is', 'igbo',
  'ig', 'indonesian', 'id', 'irish', 'ga', 'italian', 'it', 
     'japanese', 'ja', 'javanese', 'jw', 'kannada', 'kn',
  'kazakh', 'kk', 'khmer', 'km', 'korean', 'ko', 'kurdish (kurmanji)',
     'ku', 'kyrgyz', 'ky', 'lao', 'lo', 
  'latin', 'la', 'latvian', 'lv', 'lithuanian', 'lt', 'luxembourgish',
     'lb', 'macedonian', 'mk', 'malagasy',
  'mg', 'malay', 'ms', 'malayalam', 'ml', 'maltese', 'mt', 'maori',
     'mi', 'marathi', 'mr', 'mongolian', 'mn',
  'myanmar (burmese)', 'my', 'nepali', 'ne', 'norwegian', 'no',
     'odia', 'or', 'pashto', 'ps', 'persian',
   'fa', 'polish', 'pl', 'portuguese', 'pt', 'punjabi', 'pa',
     'romanian', 'ro', 'russian', 'ru', 'samoan',
   'sm', 'scots gaelic', 'gd', 'serbian', 'sr', 'sesotho', 
     'st', 'shona', 'sn', 'sindhi', 'sd', 'sinhala',
   'si', 'slovak', 'sk', 'slovenian', 'sl', 'somali', 'so', 
     'spanish', 'es', 'sundanese', 'su', 
  'swahili', 'sw', 'swedish', 'sv', 'tajik', 'tg', 'tamil',
     'ta', 'telugu', 'te', 'thai', 'th', 'turkish', 'tr',
  'ukrainian', 'uk', 'urdu', 'ur', 'uyghur', 'ug', 'uzbek', 
     'uz', 'vietnamese', 'vi', 'welsh', 'cy', 'xhosa', 'xh',
  'yiddish', 'yi', 'yoruba', 'yo', 'zulu', 'zu')
###
st.info('Application is Running wait for 2 Minutes...')
##
###Load the data set
dataset_path='C:\\Users\\DSHARATH\\Downloads\\Audio_Sentiment_Analysis\\TESS_Toronto_emotional_speech_set_data'
paths = []
labels = []
for dirname, _, filenames in os.walk(dataset_path):
    for filename in filenames:
        paths.append(os.path.join(dirname, filename))
        label = filename.split('_')[-1]
        label = label.split('.')[0]
        labels.append(label.lower())
    if len(paths) == 2800:
        break

#st.write('Dataset is Loaded')
## Create a dataframe
df = pd.DataFrame()
df['speech'] = paths
df['label'] = labels
#df.head()
#Audio waveplots
def waveplot(data, sr, emotion):
    plt.figure(figsize=(10,4))
    plt.title(emotion, size=20)
    librosa.display.waveshow(data, sr=sr)
    plt.show()
    mfcc = librosa.feature.mfcc(y=data, sr=sr, hop_length=512, n_mfcc=13)
    #print(mfcc)
    
def spectogram(data, sr, emotion):
    x = librosa.stft(data)
    xdb = librosa.amplitude_to_db(abs(x))
    plt.figure(figsize=(11,4))
    plt.title(emotion, size=20)
    librosa.display.specshow(xdb, sr=sr, x_axis='time', y_axis='hz')
    plt.colorbar()

emotion = 'angry'  # repeat here for every emotion
#filename=librosa.example(df['speech'][0])
path=df['speech'][0] ## add your emotion dataframe here
#path = np.array(df['speech'][df['label']==emotion])[0]
data, sampling_rate = librosa.load(path)
waveplot(data, sampling_rate, emotion)
spectogram(data, sampling_rate, emotion)
#st.write(sampling_rate)

#Feature Extraction
def extractFeature(filename,filepath,mfcc,chroma,mel):
    #with sf.SoundFile(filename) as soundfile:
    #print("this is file name:",filename)
    #print("this is file path",filepath+"\\"+filename)
    ffilename=str(filepath)+"\\"+str(filename)
    #st.write("Hi,--",ffilename)
    with open(ffilename,'rb') as soundfile:
        X,sampleRate=sf.read(soundfile,dtype="float32")
        #sampleRate=soundfile.samplerate
        if chroma:
            stft=np.abs(librosa.stft(X))
            result=np.array([])
        if mfcc:
            mfcc=np.mean(librosa.feature.mfcc(y=X,sr=sampleRate,n_mfcc=40).T,axis=0)
            result=np.hstack((result,mfcc))
        if chroma:
            chroma=np.mean(librosa.feature.chroma_stft(S=stft,sr=sampleRate).T,axis=0)
            result=np.hstack((result,chroma))
        if mel:
            mel=np.mean(librosa.feature.melspectrogram(y=X,sr=sampleRate).T,axis=0)
            result=np.hstack((result,mel))
        
        #sf.SoundFile.close(soundfile)
    return result

#considering emotions
emotions={'angry':'angry','disgust':'disgust','fear':'fear','happy':'happy','neutral':'neutral','ps':'pleasunt_surprise','sad':'sad'}
observedEmotions=['angry','sad','happy','fear','disgust','neutral','ps']


dataset_path='C:\\Users\\DSHARATH\\Downloads\\Audio_Sentiment_Analysis\\TESS_Toronto_emotional_speech_set_data'
x = []
y = []
def load_data(test_size=0.2):
    for dirname, _, filenames in os.walk(dataset_path):
        for filename in filenames:
            fileName=os.path.basename(filename)
            #print(filename)
            #print("this is dir name-->",dirname)
            #print("this is path",dataset_path)
            emotion=fileName.split('_')[-1]
            #print(emotion)
            emotion=emotion.split('.')[0]
            emotion=emotion.lower()
            #print(emotion)#==angry
            emotion1=emotions[emotion]
            if emotion1 not in observedEmotions:
                continue
            feature=extractFeature(fileName,dirname,mfcc=True,chroma=True,mel=True)
            x.append(feature)
            y.append(emotion1)
    return train_test_split(np.array(x),np.array(y),test_size=test_size,random_state=9)
       

#st.write('train test split is done')

##splitting dataset
xTrain,xTest,yTrain,yTest=load_data(test_size=0.2)
##getting the shape of the training and testing datasets
#st.write((xTrain.shape[0],xTest.shape[0]))

##getting the number of feature extracted
#st.write(f'Features extracted:{xTrain.shape[1]}')

##SVM Model
model_SVM=SVC()
model_SVM.fit(xTrain, yTrain)
#Predict the response for test dataset
y_pred_SVM = model_SVM.predict(xTest)

Pkl_Filename_SVM = "Emotion_Voice_Detection_Model_SVM.pkl"  

with open(Pkl_Filename_SVM, 'wb') as file_SVM:  
    pickle.dump(model_SVM, file_SVM)
# Load the Model back from file
with open(Pkl_Filename_SVM, 'rb') as file_SVM:  
    Emotion_Voice_Detection_Model_SVM = pickle.load(file_SVM)
    
# Model Accuracy, how often is the classifier correct?
st.write("Accuracy:",metrics.accuracy_score(yTest, y_pred_SVM))
#Emotion_Voice_Detection_Model_SVM
#cross validation
#cross validation for SVM model
k_folds_SVM = KFold(n_splits = 5)
scores_SVM = cross_val_score(model_SVM, xTrain, yTrain, cv = k_folds_SVM)
#print("Cross Validation Scores: ", scores_SVM)
#print("Average CV Score: ", scores_SVM.mean())
st.write("Number of CV Scores used in Average: ", len(scores_SVM))
st.info("Please Select Your Option from Left")
time.sleep(10)
###
def uploaded_audio():
    # upload file
    # Create a file uploader for WAV audio files
    audio_file = st.file_uploader("Choose a WAV audio file", type=["wav"])
    time.sleep(10)
    #extract features of an uploaded audio file
    def extractFeature_Upload(filename,mfcc,chroma,mel):
        #with sf.SoundFile(filename) as soundfile:
        #print("this is file name:",filename)
        #print("this is file path",filepath+"\\"+filename)
        #ffilename=str(filepath)+"\\"+str(filename)
        #st.write("Hi,--",ffilename)
        #with open(ffilename,'rb') as soundfile:
    
        X,sampleRate=sf.read(filename,dtype="float32")
        #sampleRate=soundfile.samplerate
        if chroma:
            stft=np.abs(librosa.stft(X))
            result=np.array([])
        if mfcc:
            mfcc=np.mean(librosa.feature.mfcc(y=X,sr=sampleRate,n_mfcc=40).T,axis=0)
            result=np.hstack((result,mfcc))
        if chroma:
            chroma=np.mean(librosa.feature.chroma_stft(S=stft,sr=sampleRate).T,axis=0)
            result=np.hstack((result,chroma))
        if mel:
            mel=np.mean(librosa.feature.melspectrogram(y=X,sr=sampleRate).T,axis=0)
            result=np.hstack((result,mel))
        
        #sf.SoundFile.close(soundfile)
        return result
        ###
    #audio_filepath=str(os.getcwd())
    live_audio_new_feature = extractFeature_Upload(audio_file,mfcc=True, chroma=True, mel=True)
    ans = np.array(live_audio_new_feature)
    #testing
    pridicted_emotion=Emotion_Voice_Detection_Model_SVM.predict([ans])
    st.success(pridicted_emotion[0])
    ###Translate
    st.subheader("Welcome to language Translate...!")
    st.text("Which language you want to convert...")
    # first argument takes the box title
    # second argument takes the options to show
    dest_language = st.selectbox("Select Language: ",
                         ['Telugu', 'Hindi', 'English','Marathi','Malayalam','Kannada','Bengali'],key=2)
    dest_language=dest_language.lower()
    # write the selected options
    st.success(dest_language)
    while (dest_language not in dic):
        st.info("Language in which you are trying to convert is currently not available ,please input some other language")

    text_to_translate = GoogleTranslator(source="auto", target=dest_language).translate(audio_file)
    st.success(text_to_translate)
    speak = gTTS(text=text_to_translate, lang=dest_language, slow=False)
  
    # Using save() method to save the translated
    # speech in capture_voice.mp3
    speak.save("captured_voice.wav")
    audio_file = open('captured_voice.wav', 'rb')
    audio_bytes = audio_file.read()
    st.audio(audio_bytes, format='audio/ogg')

    sample_rate = 44100  # 44100 samples per second
    seconds = 1  # Note duration of 2 seconds
    frequency_la = 440  # Our played note will be 440 Hz
    # Generate array with seconds*sample_rate steps, ranging between 0 and seconds
    t = np.linspace(0, seconds, seconds * sample_rate, False)
    # Generate a 440 Hz sine wave
    note_la = np.sin(frequency_la * t * 2 * np.pi)
    
#####
def live_recording():
    CHUNK = 1024 
    FORMAT = pyaudio.paInt16 #paInt8
    CHANNELS = 1 
    RATE = 44100 #sample rate
    RECORD_SECONDS = 10
    WAVE_OUTPUT_FILENAME_ORIGINAL = "live_recording.wav"
    WAVE_OUTPUT_FILENAME_MODIFIED = "Modified_recording.wav"
    p = pyaudio.PyAudio()#initilization
    stream = p.open(format=FORMAT,channels=CHANNELS,rate=RATE,input=True,frames_per_buffer=CHUNK) #buffer
    st.info("Record Your voice With in 10 Seconds")
    st.info("Recording Started")
    frames = []

    for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        frames.append(data) # 2 bytes(16 bits) per channel

    st.info("Recording Completed")
    stream.stop_stream()
    stream.close()
    p.terminate()
    wf = wave.open(WAVE_OUTPUT_FILENAME_ORIGINAL, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()
    ###Creating modified audio for re use
    p = pyaudio.PyAudio()#initilization
    stream = p.open(format=FORMAT,channels=CHANNELS,rate=RATE,input=True,frames_per_buffer=CHUNK) #buffer
    #st.info("* recording/creating modified audio file")
    frames = []
    for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        frames.append(data) # 2 bytes(16 bits) per channel

        #st.info("* done recording/created modified audio file for reuse")
    stream.stop_stream()
    stream.close()
    p.terminate()
    wf = wave.open(WAVE_OUTPUT_FILENAME_MODIFIED, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()
    ## Appying extract_feature function on random file and then loading model to predict the result  
    audio_file = 'live_recording.wav'
    audio_filepath=str(os.getcwd())
    live_audio_new_feature = extractFeature(audio_file,audio_filepath,mfcc=True, chroma=True, mel=True)
    ans = np.array(live_audio_new_feature)
    ##Testing recoreded audio
    pridicted_emotion=Emotion_Voice_Detection_Model_SVM.predict([ans])
    st.success(pridicted_emotion[0])
    # Translating from src to dest
    st.subheader("Welcome to language Translate...!")
    st.text("Which language you want to convert...")
    # first argument takes the box title
    # second argument takes the options to show
    dest_language = st.selectbox("Select Language: ",
                         ['Telugu', 'Hindi', 'English','Marathi','Malayalam','Kannada','Bengali'],key=2)
    dest_language=dest_language.lower()
    # write the selected options
    st.success(dest_language)
    while (dest_language not in dic):
        st.info("Language in which you are trying to convert is currently not available ,please input some other language")

    text_to_translate = GoogleTranslator(source="auto", target=dest_language).translate(audio_file)
    st.success(text_to_translate)
    speak = gTTS(text=text_to_translate, lang=dest_language, slow=False)
  
    # Using save() method to save the translated
    # speech in capture_voice.mp3
    speak.save("captured_voice.wav")
    audio_file = open('captured_voice.wav', 'rb')
    audio_bytes = audio_file.read()
    st.audio(audio_bytes, format='audio/ogg')

    sample_rate = 44100  # 44100 samples per second
    seconds = 1  # Note duration of 2 seconds
    frequency_la = 440  # Our played note will be 440 Hz
    # Generate array with seconds*sample_rate steps, ranging between 0 and seconds
    t = np.linspace(0, seconds, seconds * sample_rate, False)
    # Generate a 440 Hz sine wave
    note_la = np.sin(frequency_la * t * 2 * np.pi)
####
with st.sidebar:
    option = st.radio(" SELECT YOUR OPTION",[" ","Audio Sentiment Analysis","Tone Modification"],index=0)
    


if (option=="Audio Sentiment Analysis"):
    Audio_Selection = st.selectbox("Select Audio Type Here: ",
                     ["Upload From Local Device", "Live Recording"],key=1)
    if (Audio_Selection=="Upload From Local Device"):
        uploaded_audio()
    elif (Audio_Selection=="Live Recording"):
        if st.button('Click here to Record Your Voice'):
            with st.spinner('Running Microphone'):
                #RECORDED USING MICROPHONE
                live_recording()  
##





