import pandas as pd
import os

# Define the directory containing the Excel files
directory = 'Dev'  # Replace with the actual path to your directory

# Define the updated mapping
emotion_to_valence_arousal = {
    'fear': {'valence': 0, 'arousal': 1},
    'sadness': {'valence': 0, 'arousal': 0},
    'joy': {'valence': 1, 'arousal': 1},
    'disgust': {'valence': 0, 'arousal': 1},
    'anger': {'valence': 0, 'arousal': 1}
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
    df['valence'] = df['Emotion'].map(lambda x: emotion_to_valence_arousal[x]['valence'])
    df['arousal'] = df['Emotion'].map(lambda x: emotion_to_valence_arousal[x]['arousal'])

    # Save the updated dataset back to Excel
    df.to_csv(file_path, index=False)
    print(f"Processed file: {file_path}")


# Process all Excel files in the directory
for filename in os.listdir(directory):
    if filename.endswith('.csv'):
        file_path = os.path.join(directory, filename)
        process_file(file_path)

print("Processing complete.")
