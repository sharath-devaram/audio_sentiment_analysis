import streamlit as st
import librosa
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline



# Load the pre-trained model
model = make_pipeline(
    StandardScaler(),
    PCA(n_components=100),
    SVC(kernel='rbf', C=10, gamma=0.01, probability=True)
)
model.fit(X_train, y_train)



# Define the Streamlit app
st.title("Audio Sentiment Analysis")



# Create a file uploader widget
uploaded_file = st.file_uploader("Drag and drop an audio file here or click to browse", type=["wav", "mp3"])



# If a file is uploaded, extract features and make predictions
if uploaded_file is not None:
    # Load the audio file and extract features
    audio, sr = librosa.load(uploaded_file)
    mfccs = librosa.feature.mfcc(audio, sr=sr, n_mfcc=40)
    features = np.mean(mfccs.T, axis=0)



    # Make predictions using the pre-trained model
    prediction = model.predict([features])[0]
    proba = model.predict_proba([features])[0]



# Display the sentiment prediction and probability
    if prediction == 1:
        st.write("Sentiment: angry")
    elif prediction == 2:
        st.write("Sentiment: sad")
    elif prediction == 3:
        st.write("Sentiment: happy")
    elif prediction == 4:
        st.write("Sentiment: fear")
    elif prediction == 5:
        st.write("Sentiment: disgust")
    elif prediction == 6:
        st.write("Sentiment: neutral")
    else:
        st.write("Sentiment: ps")
    st.write("Probability: {:.2f}%".format(proba.max() * 100))