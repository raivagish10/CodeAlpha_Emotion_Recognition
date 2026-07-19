import numpy as np
import librosa
from tensorflow.keras.models import load_model

# ==========================================
# LOAD TRAINED MODEL
# ==========================================
model = load_model("emotion_model.h5")

# ==========================================
# EMOTION LABELS
# Must match the order used during training
# ==========================================
emotion_labels = [
    'angry',
    'calm',
    'disgust',
    'fearful',
    'happy',
    'neutral',
    'sad',
    'surprised'
]

# ==========================================
# FEATURE EXTRACTION
# ==========================================
def extract_features(file_path):
    try:
        audio, sample_rate = librosa.load(
            file_path,
            duration=3,
            offset=0.5
        )

        mfcc = librosa.feature.mfcc(
            y=audio,
            sr=sample_rate,
            n_mfcc=40
        )

        mfcc = np.mean(mfcc.T, axis=0)

        return mfcc

    except Exception as e:
        print("Error loading file:")
        print(e)
        return None


# ==========================================
# ENTER AUDIO FILE PATH
# ==========================================
audio_file = input("Enter audio file path: ")

# ==========================================
# EXTRACT FEATURES
# ==========================================
feature = extract_features(audio_file)

if feature is None:
    print("Unable to process audio file.")
    exit()

feature = feature.reshape(1, -1)

# ==========================================
# PREDICT EMOTION
# ==========================================
prediction = model.predict(feature, verbose=0)

predicted_index = np.argmax(prediction)
predicted_emotion = emotion_labels[predicted_index]
confidence = np.max(prediction) * 100

# ==========================================
# DISPLAY RESULT
# ==========================================
print("\n===== PREDICTION RESULT =====")
print("Predicted Emotion :", predicted_emotion)
print("Confidence        : {:.2f}%".format(confidence))
