import time
import pandas as pd
from tqdm import tqdm
from emotion_detection_utility.data_loader import load_encoding, load_datasheet
from emotion_detection_utility.video_processing import process_video
from emotion_detection_utility.config import EXCEl_PATH

# Load known faces and embeddings
print("Loading encodings...")
data = load_encoding()

# Load the CSV file containing video metadata
df = load_datasheet()
df['Predicted Emotion'] = None

# Process each video
for index, row in df.iterrows():
    start_time = time.time()
    print(f"\nProcessing Dialogue_ID {row['Dialogue_ID']} and Utterance_ID {row['Utterance_ID']}.")
    overall_emotion_freq, overall_emotion = process_video(row, data, df)
    if overall_emotion_freq is not None:
        df.at[index, 'Predicted Emotion'] = overall_emotion
    print(f"Actual Emotion: {row['Emotion']}, Predicted Emotion: {overall_emotion}.")
    end_time = time.time()
    print(f"Time taken to process: {(end_time - start_time) / 60:.2f} minutes.")

# Save the updated DataFrame back to the CSV file
df.to_excel(EXCEl_PATH, index=False)
print("File updated with detected emotions.")

print("Completed!")
