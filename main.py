import time
import pandas as pd
from tqdm import tqdm
from emotion_detection.data_loader import load_encoding, load_datasheet
from emotion_detection.video_processing import process_video
from emotion_detection.config import EXCEl_PATH

start_time = time.time()
# Load known faces and embeddings
print("Loading encodings...")
data = load_encoding()

# Load the CSV file containing video metadata
df = load_datasheet()
df['result_1'] = None
df['result_2'] = None

# overall_progress = tqdm(total=len(df), desc="Processing videos", unit="video")

# Process each video
for index, row in df.iterrows():
    print(f"Processing {index}th video.")
    overall_emotion_freq, overall_emotion = process_video(row, data, df)
    if overall_emotion_freq is not None:
        df.at[index, 'result_1'] = overall_emotion_freq
        df.at[index, 'result_2'] = overall_emotion
    # overall_progress.update(1)

# overall_progress.close()

# Save the updated DataFrame back to the CSV file
df.to_excel(EXCEl_PATH, index=False)
print("CSV file updated with detected emotions.")
end_time = time.time()
time_taken = (end_time-start_time)/60
print(f"Time Taken: {time_taken:.2f} minutes.")


