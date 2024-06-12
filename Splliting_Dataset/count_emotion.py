import pandas as pd
from collections import Counter

# Load your dataset
dataset_path = 'train_sent_emo.csv'
data = pd.read_csv(dataset_path)

# Assuming the emotion column is named 'emotion', adjust if necessary
emotion_column = 'Emotion'

# Count the occurrences of each emotion
emotion_counts = Counter(data[emotion_column])

# Print the counts for each emotion
for emotion, count in emotion_counts.items():
    print(f"{emotion}: {count}")
