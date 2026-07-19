
import os

DATASET_PATH = r"C:\Users\91972\Downloads\Audio_Speech_Actors_01-24 (2)\Audio_Speech_Actors_01-24"

print(DATASET_PATH)
print(os.path.exists(DATASET_PATH))
import numpy as np
import pandas as pd
import librosa
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# ==============================
# DATASET PATH
# ==============================
DATASET_PATH = r"C:\Users\91972\Downloads\Audio_Speech_Actors_01-24 (2)\Audio_Speech_Actors_01-24"

# ==============================
# RAVDESS Emotion Mapping
# ==============================
emotion_dict = {
    '01': 'neutral',
    '02': 'calm',
    '03': 'happy',
    '04': 'sad',
    '05': 'angry',
    '06': 'fearful',
    '07': 'disgust',
    '08': 'surprised'
}

# ==============================
# FEATURE EXTRACTION USING MFCC
# ==============================
def extract_features(file_path):
    try:
        audio, sample_rate = librosa.load(file_path, duration=3, offset=0.5)

        mfcc = librosa.feature.mfcc(
            y=audio,
            sr=sample_rate,
            n_mfcc=40
        )

        mfcc_scaled = np.mean(mfcc.T, axis=0)

        return mfcc_scaled

    except Exception as e:
        print("Error:", file_path)
        return None


# ==============================
# LOAD DATASET
# ==============================
X = []
Y = []

for root, dirs, files in os.walk(DATASET_PATH):

    for file in files:

        if file.endswith(".wav"):

            parts = file.split('-')

            emotion = emotion_dict[parts[2]]

            file_path = os.path.join(root, file)

            feature = extract_features(file_path)

            if feature is not None:
                X.append(feature)
                Y.append(emotion)

X = np.array(X)
Y = np.array(Y)

print("Total Samples:", len(X))
print("Feature Shape:", X.shape)

# ==============================
# LABEL ENCODING
# ==============================
encoder = LabelEncoder()

y = encoder.fit_transform(Y)
y = to_categorical(y)

print("Classes:", encoder.classes_)

# ==============================
# TRAIN TEST SPLIT
# ==============================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# ==============================
# BUILD MODEL
# ==============================
model = Sequential()

model.add(Dense(256,
                activation='relu',
                input_shape=(40,)))

model.add(Dropout(0.3))

model.add(Dense(128,
                activation='relu'))

model.add(Dropout(0.3))

model.add(Dense(64,
                activation='relu'))

model.add(Dense(y.shape[1],
                activation='softmax'))

# ==============================
# COMPILE MODEL
# ==============================
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# ==============================
# TRAIN MODEL
# ==============================
history = model.fit(
    X_train,
    y_train,
    epochs=100,
    batch_size=32,
    validation_data=(X_test, y_test)
)

# ==============================
# EVALUATE MODEL
# ==============================
loss, accuracy = model.evaluate(X_test, y_test)

print("\nAccuracy:", accuracy*100)

# ==============================
# PREDICTIONS
# ==============================
predictions = model.predict(X_test)

y_pred = np.argmax(predictions, axis=1)
y_true = np.argmax(y_test, axis=1)

print("\nClassification Report")
print(classification_report(
    y_true,
    y_pred,
    target_names=encoder.classes_
))

# ==============================
# CONFUSION MATRIX
# ==============================
cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(10,8))

sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    xticklabels=encoder.classes_,
    yticklabels=encoder.classes_
)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# ==============================
# ACCURACY GRAPH
# ==============================
plt.figure(figsize=(8,5))

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])

plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')

plt.legend([
    'Train',
    'Validation'
])

plt.show()

# ==============================
# SAVE MODEL
# ==============================
model.save("emotion_model.h5")

print("\nModel Saved Successfully")
