import pandas as pd
import os

# Constants and paths
PATH_MIXED_DATASET = "../Raw Data/train_sent_emo.csv"
OUTPUT_DIR = "../Speakers/Train"

# Load the dataset
df = pd.read_csv(PATH_MIXED_DATASET)

# Define the main 6 speakers
main_speakers = ["Monica", "Joey", "Ross", "Rachel", "Phoebe", "Chandler"]

# Create a directory to save the files if it doesn't exist
output_dir = OUTPUT_DIR
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Loop through each main speaker and create a separate DataFrame and save it as a CSV file
for speaker in main_speakers:
    speaker_df = df[df['Speaker'] == speaker]
    file_path = os.path.join(output_dir, f"{speaker}.csv")
    speaker_df.to_csv(file_path, index=False)
    print(f"Saved {speaker}'s data to {file_path}")

# Create a DataFrame for all other speakers and save it as a CSV file
other_speakers_df = df[~df['Speaker'].isin(main_speakers)]
other_speakers_file_path = os.path.join(output_dir, "other_speakers.csv")
other_speakers_df.to_csv(other_speakers_file_path, index=False)
print(f"Saved other speakers' data to {other_speakers_file_path}")