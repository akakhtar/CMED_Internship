import pandas as pd
import os

# Constants and paths
PATH_MIXED_DATASET = "../Dataset/dev_sent_emo.csv"
OUTPUT_DIR = "valence_model/Dev"

# Load the dataset
df = pd.read_csv(PATH_MIXED_DATASET)

# Define the main 6 speakers
main_speakers = ["Monica", "Joey", "Ross", "Rachel", "Phoebe", "Chandler"]

# Create a directory to save the files if it doesn't exist
output_dir = OUTPUT_DIR
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Define the updated mapping
emotion_to_valency_arousal = {
    'fear': {'valency': 0, 'arousal': 1},
    'sadness': {'valency': 0, 'arousal': 0},
    'joy': {'valency': 1, 'arousal': 1},
    'disgust': {'valency': 0, 'arousal': 1},
    'anger': {'valency': 0, 'arousal': 1}
}

# List of emotions to ignore
ignore_emotions = ['neutral', 'surprise']

# Function to process a single file
def process_file(file_path):
    # Load the Excel file
    df = pd.read_csv(file_path)

    # Filter out neutral and surprise emotions
    df = df[~df['Emotion'].isin(ignore_emotions)]

    # Create new columns for valency and arousal
    df['valency'] = df['Emotion'].map(lambda x: emotion_to_valency_arousal[x]['valency'])
    df['arousal'] = df['Emotion'].map(lambda x: emotion_to_valency_arousal[x]['arousal'])

    # Save the updated dataset back to CSV
    df.to_csv(file_path, index=False)
    print(f"Processed file: {file_path}")

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
