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
    mfccs_mean = np.mean(mfccs, axis=1).tolist()

    # Extracting Mel Spectrogram
    mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr)
    mel_spectrogram_mean = np.mean(mel_spectrogram, axis=1).tolist()

    # Extracting Spectral Contrast
    spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    spectral_contrast_mean = np.mean(spectral_contrast, axis=1).tolist()

    return mfccs_mean, mel_spectrogram_mean, spectral_contrast_mean

def audio_analysis_file(video_folder, csv_file):
    df = pd.read_csv(csv_file)

    # Initialize columns for the features
    f1 = {}
    f2 = {}
    f3 = {}

    for index, row in df.iterrows():
        dialogue_id = row['Dialogue_ID']
        utterance_id = row['Utterance_ID']
        video_file = f"dia{dialogue_id}_utt{utterance_id}.mp4"
        video_path = os.path.join(video_folder, video_file)
        print(f"Processing audio: {video_file}")

        if not os.path.exists(video_path):
            print(f"Video file {video_file} not found.")
            continue

        try:
            audio = extract_audio(video_path)
            mfccs_mean, mel_spectrogram_mean, spectral_contrast_mean = extract_audio_features(audio)
            f1[index] = mfccs_mean
            f2[index] = mel_spectrogram_mean
            f3[index] = spectral_contrast_mean

        except Exception as e:
            print(f"Error processing video {video_file}: {e}")

    df['mfccs'] = pd.Series(f1)
    df['melSpectrogram'] = pd.Series(f2)
    df['spectralContrast'] = pd.Series(f3)


    # Save updated DataFrame back to CSV
    df.to_csv(csv_file, index=False)
    print("CSV file updated")

video_folder = "../../Dataset/Dev/"
csv_file = '../Speakers/Dev/Monica_dev.csv'
startTime = time.time()
audio_analysis_file(video_folder, csv_file)
endTime = time.time()
print(f"Total time: {endTime-startTime} seconds")