
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
#from sklearn.tree import DecisionTreeClassifier
#from sklearn.ensemble import RandomForestClassifier
###xgb
#from sklearn.preprocessing import LabelEncoder
#from keras.utils import np_utils
#from xgboost import XGBClassifier
##
#import plotly.graph_objects as go
from audio_recorder_streamlit import audio_recorder
from playsound import playsound
#import speech_recognition as sr 
#from googletrans import Translator 
#from gtts import gTTS 
import time
#from deep_translator import GoogleTranslator
import streamlit as st
import warnings
warnings.filterwarnings('ignore')
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

set_background('C:\\Users\\DSHARATH\\Downloads\\Sentiment_Analysis\\static\\Page11.png')
st.title("Audio Sentiment Analysis")
#st.warning("Application is running, please wait, ETA: 2 Minutes")
###Load the data set
#dataset_path='C:\\Users\\DSHARATH\\Downloads\\Audio_Sentiment_Analysis\\TESS_Toronto_emotional_speech_set_data'
#paths = []
#labels = []
#for dirname, _, filenames in os.walk(dataset_path):
    #for filename in filenames:
        #paths.append(os.path.join(dirname, filename))
        #label = filename.split('_')[-1]
        #label = label.split('.')[0]
        #labels.append(label.lower())
    #if len(paths) == 2800:
        #break

#st.write('Dataset is Loaded')
## Create a dataframe
#df = pd.DataFrame()
#df['speech'] = paths
#df['label'] = labels
#df.head()
#Audio waveplots
#def waveplot(data, sr, emotion):
    #plt.figure(figsize=(10,4))
    #plt.title(emotion, size=20)
    #librosa.display.waveshow(data, sr=sr)
    #plt.show()
    #mfcc = librosa.feature.mfcc(y=data, sr=sr, hop_length=512, n_mfcc=13)
    #print(mfcc)
    
#def spectogram(data, sr, emotion):
    #x = librosa.stft(data)
    #xdb = librosa.amplitude_to_db(abs(x))
    #plt.figure(figsize=(11,4))
    #plt.title(emotion, size=20)
    #librosa.display.specshow(xdb, sr=sr, x_axis='time', y_axis='hz')
    #plt.colorbar()

###emotion = 'angry'  # repeat here for every emotion ### for editing
#filename=librosa.example(df['speech'][0])
###path=df['speech'][0] ## add your emotion dataframe here  ###for editing
#path = np.array(df['speech'][df['label']==emotion])[0]
###data, sampling_rate = librosa.load(path)   ###for editing
###waveplot(data, sampling_rate, emotion)  ###for editing
###spectogram(data, sampling_rate, emotion) ###for editing
#st.write(sampling_rate)

#Feature Extraction
#def extractFeature(filename,filepath,mfcc,chroma,mel):
    #with sf.SoundFile(filename) as soundfile:
    #print("this is file name:",filename)
    #print("this is file path",filepath+"\\"+filename)
    #ffilename=str(filepath)+"\\"+str(filename)
    #st.write("Hi,--",ffilename)
    #with open(ffilename,'rb') as soundfile:
        #X,sampleRate=sf.read(soundfile,dtype="float32")
        #sampleRate=soundfile.samplerate
        #if chroma:
            #stft=np.abs(librosa.stft(X))
            #result=np.array([])
        #if mfcc:
            #mfcc=np.mean(librosa.feature.mfcc(y=X,sr=sampleRate,n_mfcc=40).T,axis=0)
            #result=np.hstack((result,mfcc))
        #if chroma:
            #chroma=np.mean(librosa.feature.chroma_stft(S=stft,sr=sampleRate).T,axis=0)
            #result=np.hstack((result,chroma))
        #if mel:
            #mel=np.mean(librosa.feature.melspectrogram(y=X,sr=sampleRate).T,axis=0)
            #result=np.hstack((result,mel))
        
        #sf.SoundFile.close(soundfile)
    #return result

#considering emotions
#emotions={'angry':'angry','disgust':'disgust','fear':'fear','happy':'happy','neutral':'neutral','ps':'pleasunt_surprise','sad':'sad'}
#observedEmotions=['angry','sad','happy','fear','disgust','neutral','ps']


#dataset_path='C:\\Users\\DSHARATH\\Downloads\\Audio_Sentiment_Analysis\\TESS_Toronto_emotional_speech_set_data'
#x = []
#y = []
#def load_data(test_size=0.2):
    #for dirname, _, filenames in os.walk(dataset_path):
        #for filename in filenames:
            #fileName=os.path.basename(filename)
            #print(filename)
            #print("this is dir name-->",dirname)
            #print("this is path",dataset_path)
            #emotion=fileName.split('_')[-1]
            #print(emotion)
            #emotion=emotion.split('.')[0]
            #emotion=emotion.lower()
            #print(emotion)#==angry
            #emotion1=emotions[emotion]
            #if emotion1 not in observedEmotions:
                #continue
            #feature=extractFeature(fileName,dirname,mfcc=True,chroma=True,mel=True)
            #x.append(feature)
            #y.append(emotion1)
    #return train_test_split(np.array(x),np.array(y),test_size=test_size,random_state=9)
       

#st.write('train test split is done')

##splitting dataset
#with st.spinner("Application is running"):
    #xTrain,xTest,yTrain,yTest=load_data(test_size=0.2)
##getting the shape of the training and testing datasets
#st.write((xTrain.shape[0],xTest.shape[0]))

##getting the number of feature extracted
#st.write(f'Features extracted:{xTrain.shape[1]}')
#with st.spinner("Application is running"):
    ##SVM Model
    #model_SVM=SVC()
    #model_SVM.fit(xTrain, yTrain)
    #Predict the response for test dataset
    #y_pred_SVM = model_SVM.predict(xTest)

    #Pkl_Filename_SVM = "Emotion_Voice_Detection_Model_SVM.pkl"  

#with open(Pkl_Filename_SVM, 'wb') as file_SVM:  
    #pickle.dump(model_SVM, file_SVM)
# Load the Model back from file
with st.spinner("Application is running"):
    Pkl_Filename_SVM = "Emotion_Voice_Detection_Model_SVM.pkl" 
    with open(Pkl_Filename_SVM, 'rb') as file_SVM:  
        Emotion_Voice_Detection_Model_SVM = pickle.load(file_SVM)
    
# Model Accuracy, how often is the classifier correct?
#st.write("Accuracy:",metrics.accuracy_score(yTest, y_pred_SVM))
#Emotion_Voice_Detection_Model_SVM
#cross validation
#cross validation for SVM model
#k_folds_SVM = KFold(n_splits = 5)
#scores_SVM = cross_val_score(model_SVM, xTrain, yTrain, cv = k_folds_SVM)
#print("Cross Validation Scores: ", scores_SVM)
#print("Average CV Score: ", scores_SVM.mean())
#st.write("Number of CV Scores used in Average: ", len(scores_SVM))

#####
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
####
def live_recording():
    st.info("Connecting Microphone...")
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
    with st.spinner("Predicting Emotion..."):
        st.info("Feature extraction is in progress...")
        #live_audio_new_feature = extractFeature(audio_file,audio_filepath,mfcc=True, chroma=True, mel=True)
        live_audio_new_feature = extractFeature_Upload(audio_file,mfcc=True, chroma=True, mel=True)
        ans = np.array(live_audio_new_feature)
        ##Testing recoreded audio
        st.info("Predicting the Emotion....")
        pridicted_emotion=Emotion_Voice_Detection_Model_SVM.predict([ans])
    st.success("Emotion Predicted...🙂")
    st.success(pridicted_emotion[0])
####
# second argument takes options
st.success("Application is ready to use")
option = st.selectbox("Select Option Here:",['File Upload', 'Live Recording'])
#option_submit = st.button("Submit")
#if (option_submit):
# print the selected option
st.write("Selected option is : ",option)
if (option=="File Upload"):
    audio_file = st.file_uploader("Choose a WAV audio file", type=["wav"])
    st.info("Click on Upload Button After Uploading Audio File")
    predict = st.button("Upload")
    if predict:
        #if (audio_file==None):
        #raise Exception("Please Upload Audio File Only...")
        #time.sleep(5)
        #audio_filepath=str(os.getcwd())
        with st.spinner("Predicting Emotion.."):
            st.info("Feature extraction is in progress...")
            live_audio_new_feature = extractFeature_Upload(audio_file,mfcc=True, chroma=True, mel=True)
            ans = np.array(live_audio_new_feature)
            #testing
            st.info("Predicting Emotion...")
            pridicted_emotion=Emotion_Voice_Detection_Model_SVM.predict([ans])
        st.success("Emotion Predicted...🙂")
        st.success(pridicted_emotion[0])
        #audio_file = open(audio_file.name, 'rb')
        #audio_bytes = audio_file.read()
        #st.success("Listen your translated audio here...")
        #st.audio(audio_bytes, format='audio/ogg')
    else:
        st.write()
            #st.info("Please Upload File ")
elif (option=="Live Recording"):
    #st.write("Selected option is : ",option)
    live_recording()
    audio_file = open('live_recording.wav', 'rb')
    audio_bytes = audio_file.read()
    st.info("Listen your recorded voice here...🎵🎵🎵")
    st.audio(audio_bytes, format='audio/ogg')
else:
    st.write("Invalid Selection")
#else:
    #st.write("Upload failure")
#def Audio_Return():
    #return audio_file



