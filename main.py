import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import csv
import librosa
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from tensorflow import keras
import matplotlib.pyplot as plt
import librosa.display
import cv2

def extract_features(y, sr, n_mfcc=40, max_len=100):
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    if mfccs.shape[1] < max_len:
        pad_width = max_len - mfccs.shape[1]
        mfccs = np.pad(mfccs, pad_width=((0, 0), (0, pad_width)), mode='constant')
    else:
        mfccs = mfccs[:, :max_len]
    mfccs = np.expand_dims(mfccs, axis=-1)
    return mfccs

dataset = pd.read_csv("audio_dataset.csv")
X = np.array(dataset["features"].apply(eval).tolist())
y = np.array(dataset["label"])
X = X.reshape(X.shape[0], X.shape[1], X.shape[2], 1)
label_map = {'roire': 0, 'normal': 1, 'angry': 2}
y = np.array([label_map[label] for label in y])
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

model_path = "trained_model.keras"
if os.path.exists(model_path):
    model = keras.models.load_model(model_path)
    print(f"Model loaded from {model_path}")
else:
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(X_train.shape[1], X_train.shape[2], 1)),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Dropout(0.5),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(3, activation='softmax')
    ])

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=10, validation_data=(X_val, y_val))
    model.save("trained_model.keras")

input_file_path = "toot.wav"
y, sr = librosa.load(input_file_path, sr=None)
features = extract_features(y, sr)
features = np.expand_dims(features, axis=0)
prediction = model.predict(features)
predicted_class_index = np.argmax(prediction)
predicted_class_name = list(label_map.keys())[predicted_class_index]
accuracy = np.max(prediction)
print(f"Predicted class: {predicted_class_name}, Accuracy: {accuracy}")

def filter_swarm():
    y, sr = librosa.load(input_file_path)
    file_saved = "toot.png"
    init_file = "toot_initial.png"
    # Compute spectrogram
    S = librosa.stft(y)
    D = librosa.amplitude_to_db(abs(S), ref=np.max)
    # Plot spectrogram
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Spectrogram')
    plt.savefig(file_saved)
    img1 = cv2.imread(init_file)
    img2 = cv2.imread(file_saved)
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    def mse(img1, img2):
        h, w = img1.shape
        diff = cv2.subtract(img1, img2)
        err = np.sum(diff ** 2)
        mse = err / (float(h * w))
        return mse, diff
    error, diff = mse(img1, img2)
    print("Image matching Error between the two images:", error)
    if error >= 10:
        filter_correction_swarm()
    else:
        print(f"Sound is {predicted_class_name}")
        os.remove(file_saved)
def filter_normal():
    y, sr = librosa.load(input_file_path)
    file_saved = "normal.png"
    init_file = "normal_initial.png"
    # Compute spectrogram
    S = librosa.stft(y)
    D = librosa.amplitude_to_db(abs(S), ref=np.max)
    # Plot spectrogram
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Spectrogram')
    plt.savefig(file_saved)
    img1 = cv2.imread(init_file)
    img2 = cv2.imread(file_saved)
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    def mse(img1, img2):
        h, w = img1.shape
        diff = cv2.subtract(img1, img2)
        err = np.sum(diff ** 2)
        mse = err / (float(h * w))
        return mse, diff
    error, diff = mse(img1, img2)
    print("Image matching Error between the two images:", error)
    if error >= 10:
        filter_correction_normal()
    else:
        print(f"Sound is {predicted_class_name}")
        os.remove(file_saved)

def filter_correction_normal():
    y, sr = librosa.load(input_file_path)
    file_saved = "toot.png"
    init_file = "toot_initial.png"
    # Compute spectrogram
    S = librosa.stft(y)
    D = librosa.amplitude_to_db(abs(S), ref=np.max)
    # Plot spectrogram
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Spectrogram')
    plt.savefig(file_saved)
    img1 = cv2.imread(init_file)
    img2 = cv2.imread(file_saved)
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    def mse(img1, img2):
        h, w = img1.shape
        diff = cv2.subtract(img1, img2)
        err = np.sum(diff ** 2)
        mse = err / (float(h * w))
        return mse, diff

    error, diff = mse(img1, img2)
    print("Image matching Error between the two images:", error)
    if error >= 10:
        print(f"Couldn't recognize the sound")
        os.remove(file_saved)
def filter_correction_swarm():
    y, sr = librosa.load(input_file_path)
    file_saved = "normal.png"
    init_file = "normal_initial.png"
    # Compute spectrogram
    S = librosa.stft(y)
    D = librosa.amplitude_to_db(abs(S), ref=np.max)
    # Plot spectrogram
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Spectrogram')
    plt.savefig(file_saved)
    img1 = cv2.imread(init_file)
    img2 = cv2.imread(file_saved)
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    def mse(img1, img2):
        h, w = img1.shape
        diff = cv2.subtract(img1, img2)
        err = np.sum(diff ** 2)
        mse = err / (float(h * w))
        return mse, diff
    error, diff = mse(img1, img2)
    print("Image matching Error between the two images:", error)
    if error >= 10:
        print(f"Couldn't recognize the sound")
        os.remove(file_saved)
if predicted_class_name == 'roire':
    filter_swarm()
elif predicted_class_name == 'normal':
    filter_normal()