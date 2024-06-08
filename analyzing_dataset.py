# main.py
import os
import cv2
from tqdm import tqdm
import pandas as pd
from emotion_detection.data_loader import load_encoding, load_datasheet
from emotion_detection.video_analyzing import process_video
from emotion_detection.config import EXCEl_PATH, DIALOGUE_REPORT_PATH

# Load known faces and embeddings
print("Loading encodings...")
data = load_encoding()

# Load the EXCEL file containing video metadata
df = load_datasheet()
df["face_recognized"] = None

total_videos = len(df)
unrecognized_count = 0
not_in_6_char_count = 0

# Variables for dialogue-based analysis
dialogue_stats = []
dialogue_video_count = 0
dialogue_unrecognized_count = 0
dialogue_not_in_6_char_count = 0
prev_dia_id = df.iloc[0]['Dialogue_ID']  # Initialize with the first dialogue ID

# Loop through each row in the DataFrame
for index, row in df.iterrows():
    main6Char, recognized = process_video(row, data, df)

    # Update overall statistics
    if not main6Char:
        not_in_6_char_count += 1
    if not recognized:
        unrecognized_count += 1
        df.at[index, 'face_recognized'] = "NO"
    else:
        df.at[index, 'face_recognized'] = "YES"

    # Update dialogue-specific statistics
    dialogue_video_count += 1
    if not main6Char:
        dialogue_not_in_6_char_count += 1
    if not recognized:
        dialogue_unrecognized_count += 1

    curr_dia_id = row['Dialogue_ID']

    # Check if the dialogue ID has changed
    if curr_dia_id != prev_dia_id:
        # Calculate the percentage of unrecognized faces for the previous dialogue
        dialogue_unrecognized_count -= dialogue_not_in_6_char_count
        dialogue_unrecognized_percentage = (dialogue_unrecognized_count / dialogue_video_count) * 100

        # Print dialogue-specific statistics
        print(f"Dialogue ID: {prev_dia_id}")
        print(f"  Total Videos: {dialogue_video_count}")
        print(f"  Not in Main 6 Character Count: {dialogue_not_in_6_char_count}")
        print(f"  Unrecognized Count: {dialogue_unrecognized_count}")
        print(f"  Unrecognized Percentage: {dialogue_unrecognized_percentage:.2f}%")

        # Append the statistics for the completed dialogue to the list
        dialogue_stats.append({
            'Dialogue_ID': prev_dia_id,
            'Total_Videos': dialogue_video_count,
            'Not_In_Main_6_Char_Count': dialogue_not_in_6_char_count,
            'Unrecognized_Count': dialogue_unrecognized_count,
            'Unrecognized_Percentage': dialogue_unrecognized_percentage
        })

        # Reset counters for the new dialogue
        dialogue_video_count = 0
        dialogue_unrecognized_count = 0
        dialogue_not_in_6_char_count = 0
        prev_dia_id = curr_dia_id  # Update to the new dialogue ID

# Calculate the percentage of unrecognized videos for overall analysis
unrecognized_percentage = (unrecognized_count / total_videos) * 100

# Print overall statistics
print(f"Count of total videos: {total_videos}")
print(f"Count of the videos where speaker is not out of main 6 characters: {not_in_6_char_count}")
print(f"Count of videos where face was not recognized: {unrecognized_count}")
print(f"Percentage of video clips where the face is not recognized: {unrecognized_percentage:.2f}%")

# Save the overall results to the EXCEL file
df.to_csv(EXCEl_PATH, index=False)
print("CSV file updated.")

# Save the dialogue-based analysis to a new CSV file
dialogue_df = pd.DataFrame(dialogue_stats)
dialogue_df.to_excel(DIALOGUE_REPORT_PATH, index=False)
print("Dialogue-based analysis CSV file created.")
