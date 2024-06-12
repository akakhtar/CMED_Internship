import pandas as pd
import os

# Define the directory containing the Excel files
directory = 'Speakers_Dataset'  # Replace with the actual path to your directory

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
    df = pd.read_excel(file_path)

    # Filter out neutral and surprise emotions
    df = df[~df['Emotion'].isin(ignore_emotions)]

    # Create new columns for valency and arousal
    df['valency'] = df['Emotion'].map(lambda x: emotion_to_valency_arousal[x]['valency'])
    df['arousal'] = df['Emotion'].map(lambda x: emotion_to_valency_arousal[x]['arousal'])

    # Save the updated dataset back to Excel
    df.to_excel(file_path, index=False)
    print(f"Processed file: {file_path}")


# Process all Excel files in the directory
for filename in os.listdir(directory):
    if filename.endswith('.xlsx'):
        file_path = os.path.join(directory, filename)
        process_file(file_path)

print("Processing complete.")
