import numpy as np
from sklearn.preprocessing import LabelEncoder

# Define emotion labels
emotion_labels = ['happy', 'sad', 'fear', 'disgust', 'pleasant surprise', 'angry', 'neutral']

# Initialize label encoder
label_encoder = LabelEncoder()

# Fit label encoder to emotion labels
label_encoder.fit(emotion_labels)

# Save label encoder classes to numpy file
np.save('classes.py', label_encoder.classes_)
