import pandas as pd
import os

OUTPUT_DIR = "Speakers_Dataset"
main_speakers = ["Monica", "Joey", "Ross", "Rachel", "Phoebe", "Chandler"]

# Function to add scene numbers based on Dialogue_ID
def add_scene_numbers(df):
    df["scene_number"] = 0
    df = df.sort_values(by='Dialogue_ID').reset_index(drop=True)
    current_scene_number = 1
    prev_dialogue_id = df.loc[0, 'Dialogue_ID']

    for index, row in df.iterrows():
        current_dialogue_id = row['Dialogue_ID']
        if current_dialogue_id != prev_dialogue_id:
            current_scene_number += 1
            prev_dialogue_id = current_dialogue_id
        df.at[index, 'scene_number'] = current_scene_number
    return df

# Process each speaker's dataset
for speaker in main_speakers:
    file_path = os.path.join(OUTPUT_DIR, f"{speaker}.xlsx")
    if os.path.exists(file_path):
        # Load the DataFrame
        speaker_df = pd.read_excel(file_path)
        # Add scene numbers
        speaker_df = add_scene_numbers(speaker_df)
        # Save the updated DataFrame
        speaker_df.to_excel(file_path, index=False)
        print(f"Updated {speaker}'s data with scene numbers in {file_path}")

# Process the dataset for other speakers
other_speakers_file_path = os.path.join(OUTPUT_DIR, "other_speakers.xlsx")
if os.path.exists(other_speakers_file_path):
    other_speakers_df = pd.read_excel(other_speakers_file_path)
    other_speakers_df = add_scene_numbers(other_speakers_df)
    other_speakers_df.to_excel(other_speakers_file_path, index=False)
    print(f"Updated other speakers' data with scene numbers in {other_speakers_file_path}")
