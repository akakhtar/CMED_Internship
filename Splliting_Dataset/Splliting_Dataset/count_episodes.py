import pandas as pd

# Load the dataset
file_path = 'Speakers_Dataset/Ross.xlsx'  # Change to the actual file path
data = pd.read_excel(file_path)
print(f"total number of data: {data.shape}")

# Group by season and count unique episodes
season_episode_counts = data.groupby('Season')['Episode'].nunique()

# Calculate the total count of unique episodes
total_unique_episodes = data['Episode'].nunique()

# # Print the results
print("Count of unique episodes corresponding to each season:")
print(data.groupby('scene_number')['Utterance_ID'].nunique())

# print("\nTotal count of unique episodes in the dataset:")
# print(season_episode_counts.sum())
print(data['Dialogue_ID'].nunique())
