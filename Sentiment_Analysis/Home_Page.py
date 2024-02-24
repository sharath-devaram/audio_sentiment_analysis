import streamlit as st
from PIL import Image
##set page config
image = Image.open('static/spade.png')
st.set_page_config(
    page_title="Audio Sentiment Analysis",
    page_icon = image
)
image = Image.open('static/Capgemini_Logo.png')
image1 = st.sidebar.image("static/Capgemini_Logo.png", use_column_width=False, width=150)
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

set_background('C:\\Users\\DSHARATH\\Downloads\\Sentiment_Analysis\\static\\Homepage.png')
###

st.title("AUDIO SENTIMENT ANALYSIS")
st.info("🎵🎶🎵...WELCOME...🎵🎶🎵")
st.markdown(
    """
    This application is used for audio sentiment analysis.
    **👈 Start this app demo from the sidebar** 
    ### Features..
    -   You can upload your audio file 📁🎼
    -   You can record your voice here 🎙️
"""
)
st.success("Start The Application 🙂")