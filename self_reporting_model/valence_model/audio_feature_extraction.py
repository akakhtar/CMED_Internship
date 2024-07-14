import pandas as pd
from moviepy.editor import VideoFileClip
import librosa
import numpy as np
import os
import time

def extract_audio(video_path):
    video = VideoFileClip(video_path)
    audio = video.audio
    return audio

def extract_audio_features(audio):
    y = np.array(audio.to_soundarray()[:, 0])
    sr = audio.fps
    # Extracting MFCCs
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfccs_mean = np.mean(mfccs, axis=1)

    # # Extracting Chroma Features
    # chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    # chroma_mean = np.mean(chroma, axis=1)

    # Extracting Mel Spectrogram
    mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr)
    mel_spectrogram_mean = np.mean(mel_spectrogram, axis=1)

    # Extracting Spectral Contrast
    spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    spectral_contrast_mean = np.mean(spectral_contrast, axis=1)

    return mfccs_mean.mean(), mel_spectrogram_mean.mean(), spectral_contrast_mean.mean()

def audio_analysis_file(video_folder, csv_file):
    df = pd.read_csv(csv_file)

    for index, row in df.iterrows():
        dialogue_id = row['Dialogue_ID']
        utterance_id = row['Utterance_ID']
        video_file = f"dia{dialogue_id}_utt{utterance_id}.mp4"
        video_path = os.path.join(video_folder, video_file)
        print(f"Processing audio: {video_file}")

        if not os.path.exists(video_path):
            print(f"Video file {video_file} not found.")
            continue

        audio = extract_audio(video_path)
        mfccs_mean, mel_spectrogram_mean, spectral_contrast_mean = extract_audio_features(audio)

        # Update DataFrame with extracted features
        df.at[index, 'MFCCs'] = mfccs_mean
        # df.at[index, 'Chroma'] = chroma_mean
        df.at[index, 'Mel Spectrogram'] = mel_spectrogram_mean
        df.at[index, 'Spectral Contrast'] = spectral_contrast_mean

    # Save updated DataFrame back to CSV
    df.to_csv(csv_file, index=False)
    print("CSV file updated")

# Example usage
video_folder = "../Dataset/Train/"
csv_file = "Dev/Ross.csv"
startTime = time.time()
audio_analysis_file(video_folder, csv_file)
endTime = time.time()
print(f"Total time: {endTime-startTime} seconds")
