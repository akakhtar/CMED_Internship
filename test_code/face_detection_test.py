# main.py
import time
import os
import cv2
from tqdm import tqdm
import pandas as pd
from emotion_detection.data_loader import load_encoding, load_datasheet
from emotion_detection.video_analyzing import process_video
from emotion_detection.config import EXCEl_PATH, DIALOGUE_REPORT_PATH

start_time = time.time()
# Load known faces and embeddings
print("Loading encodings...")
data = load_encoding()

# Load the EXCEL file containing video metadata
df = load_datasheet()
df["Names_count_frames"] = None
df["face_recognized"] = None
df["total_frames"] = None
# Loop through each row in the DataFrame
for index, row in df.iterrows():
    print(f"Processing of the {index}th video.")
    recognized, listNames, total_frames = process_video(row, data, df)
    df.at[index, 'Names_count_frames'] = str(listNames)
    df.at[index, 'total_frames'] = total_frames
    if not recognized:
        df.at[index, 'face_recognized'] = "NO"
    else:
        df.at[index, 'face_recognized'] = "YES"

# Save the overall results to the EXCEL file
df.to_excel(EXCEl_PATH, index=False)
print("EXCEL file updated.")
end_time = time.time()
time_taken = (end_time - start_time) / 60
print(f"Time taken : {time_taken:.2f} minutes. ")
