import csv
import librosa
import numpy as np
import os

# Function to extract MFCC features and reshape for CNN input
def extract_features(y, sr, n_mfcc=40, max_len=100):
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    if mfccs.shape[1] < max_len:
        pad_width = max_len - mfccs.shape[1]
        mfccs = np.pad(mfccs, pad_width=((0, 0), (0, pad_width)), mode='constant')
    else:
        mfccs = mfccs[:, :max_len]
    mfccs = np.expand_dims(mfccs, axis=-1)  # Add channel dimension for CNN
    return mfccs

# Define directory paths and labels
dir_paths = ["audio_files/roire", "audio_files/normal", "audio_files/angry"]
labels = ["roire", "normal", "angry"]
num_copies = 100
max_len = 100  # Maximum length for features

# Extract features and create dataset
dataset = []

for dir_path, label in zip(dir_paths, labels):
    for file_name in os.listdir(dir_path):
        if file_name.endswith('.wav'):
            file_path = os.path.join(dir_path, file_name)
            for _ in range(num_copies):
                y, sr = librosa.load(file_path, sr=None)
                features = extract_features(y, sr, max_len=max_len)  # Ensure features have the same shape
                dataset.append((features, label))

# Shuffle the dataset
np.random.shuffle(dataset)

# Write dataset to CSV file
with open("a doua metoda/audio_dataset.csv", "w", newline="") as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(["features", "label"])  # Write header
    for features, label in dataset:
        csvwriter.writerow([features.tolist(), label])  # Write features as a list instead of flattening

print("Audio dataset CSV file 'audio_dataset.csv' created.")
