# main.py
import os
import cv2
from tqdm import tqdm
import pandas as pd
from emotion_detection.data_loader import load_encoding, load_datasheet
from emotion_detection.video_analyzing import process_video
from emotion_detection.config import EXCEl_PATH

# Load known faces and embeddings
print("Loading encodings...")
data = load_encoding()

# Load the EXCEL file containing video metadata
df = load_datasheet()
df["face_recognized"] = None

total_videos = len(df)
unrecognized_count = 0

# Loop through each row in the DataFrame
for index, row in df.iterrows():
    recognized = process_video(row, data, df)
    if not recognized:
        unrecognized_count += 1
        df.at[index,'face_recognized'] = "NO"
    else:
        df.at[index,'face_recognized'] = "YES"

print(f"Count of total videos: {total_videos}")
print(f"Count of videos where face was not recognized: {unrecognized_count}")
# Calculate the percentage of unrecognized videos
unrecognized_percentage = (unrecognized_count / total_videos) * 100
print(f"Percentage of video clips where the face is not recognized: {unrecognized_percentage:.2f}%")

df.to_excel(EXCEl_PATH, index=False)
print("EXCEL file updated.")