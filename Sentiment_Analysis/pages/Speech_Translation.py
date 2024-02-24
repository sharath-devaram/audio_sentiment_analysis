import streamlit as st
from pydub import AudioSegment
import numpy as np
from audio_recorder_streamlit import audio_recorder
from playsound import playsound
import speech_recognition as sr 
from googletrans import Translator 
from gtts import gTTS 
import os
import time
from deep_translator import GoogleTranslator
from os import path
#import Audio_Sentiment_Analysis as ASA
#####
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

set_background('C:\\Users\\DSHARATH\\Downloads\\Sentiment_Analysis\\static\\Page22.png')

st.title("Speech Translation")
st.success("Application is ready to use")
#st.info("Record your voice for translation...")
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

def takecommand():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        st.info("listening.....")
        r.pause_threshold = 1
        audio = r.listen(source)
  
    try:
        st.info("Recognizing.....")
        query = r.recognize_google(audio, language='en-in')
        st.success(f"user said {query}\n")
    except Exception as e:
        st.warning("say that again please.....")
        return "None"
    return query

###
def destination_language():
    st.info("Say the language in which you want to convert : Ex. Hindi , English , etc.")
    # Input destination language in which the user 
    # wants to translate
    to_lang = takecommand()
    while (to_lang == "None"):
        to_lang = takecommand()
    to_lang = to_lang.lower()
    return to_lang

# Define the function for transcribing audio
def transcribe_audio(audio_file):
    final_str = ""
    r = sr.Recognizer()
    with sr.AudioFile(audio_file) as source:
        audio = r.record(source)  # read the entire audio file                  
        #text=r.recognize_google(audio,language='en-in',show_all=True)
        text=r.recognize_google(audio,show_all=True)
        for i in range(0, len(text)-1):
            a=text["alternative"]
            for j in range(0, len(a)):
                ##
                b=a[j]
                #print(b)
                #print(type(b))
                c=b['transcript']
                #print(c)
                final_str+=c+", "
                #print(final_str)

    return final_str
####
def translated_Text(to_lang,query):
    ##
    text_to_translate = GoogleTranslator(source="auto", target=to_lang).translate(query)
    st.success(text_to_translate)
    speak = gTTS(text=text_to_translate, lang=to_lang, slow=False)
    # Using save() method to save the translated
    # speech in capture_voice.mp3
    speak.save("captured_voice.mp3")
    audio_file = open('captured_voice.mp3', 'rb')
    audio_bytes = audio_file.read()
    st.success("Listen your translated audio here...")
    st.audio(audio_bytes, format='audio/ogg')
###
####
option = st.selectbox("Select Option Here:",['File Upload','Live Recording'])
#st.info("Please wait for 10 seconds, Mic is connecting...")
#time.sleep(10)
if (option=="Live Recording"):
    st.write("Selected Option : ",option)
    # Taking voice input from the user
    query = takecommand()
    while (query == "None"):
        query = takecommand()
    to_lang = destination_language()
    # Mapping it with the code
    while (to_lang not in dic):
        st.info("Language in which you are trying to convert is currently not available ,please input some other language")
        st.write()
        to_lang = destination_language()
    to_lang = dic[dic.index(to_lang)+1]
    translated_Text(to_lang,query)
    st.success("Thank You")

elif(option=="File Upload"):
    ##
    st.write("Selected Option : ",option)
    audio_file = st.file_uploader("Upload audio file", type=["wav"])
    submit_button=st.button("Upload")
    if submit_button:
        #if audio_file:
        query = transcribe_audio(audio_file)
        st.write("Transcribed text:")
        st.write(query)
        to_lang=destination_language()
        # Mapping it with the code
        while (to_lang not in dic):
            st.write("Language in which you are trying to convert is currently not available ,please input some other language")
            st.write()
            to_lang = destination_language()
        to_lang = dic[dic.index(to_lang)+1]
        translated_Text(to_lang,query)
        st.success("Thank You")
    else:
        st.write()
else:
    st.write()


####
#audio_file=ASA.Audio_Return()

####
#sound = AudioSegment.from_wav(audio_file)
#sound = sound.set_channels(1)
#sound.export("/output/path.wav", format="wav")
# invoking Translator
# Translating from src to dest


#sample_rate = 44100  # 44100 samples per second
#seconds = 1  # Note duration of 2 seconds
#frequency_la = 440  # Our played note will be 440 Hz
# Generate array with seconds*sample_rate steps, ranging between 0 and seconds
#t = np.linspace(0, seconds, seconds * sample_rate, False)
# Generate a 440 Hz sine wave
#note_la = np.sin(frequency_la * t * 2 * np.pi)