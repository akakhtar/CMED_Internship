import pandas as pd
import numpy as np

# Load the dataset
file_path = 'Speakers_Dataset/Rachel.xlsx'  # Change to the actual file path
data = pd.read_excel(file_path)

# Initialize variables
transitions = np.zeros((2, 2))  # 2x2 matrix for valency 0 and 1
influences_0 = []
influences_1 = []

# Add columns for influence values and sequence lengths
data['Sequence_Length'] = 0
data['Influence_0'] = np.nan
data['Influence_1'] = np.nan


# Function to normalize valency to 0 or 1
def normalize_valency(val):
    return 0 if val == 0 else 1


# Iterate through each group of consecutive scene numbers
for scene in data['scene_number'].unique():
    scene_data = data[data['scene_number'] == scene].reset_index(drop=True)
    current_emotion = None
    sequence_length = 1

    # Calculate transitions and influences
    for i in range(len(scene_data)):
        curr_valency = normalize_valency(scene_data.loc[i, 'valency'])

        if i > 0:
            prev_valency = normalize_valency(scene_data.loc[i - 1, 'valency'])

            # Update transition counts
            transitions[prev_valency, curr_valency] += 1

            # Calculate time difference
            prev_time = pd.to_timedelta(scene_data.loc[i - 1, 'EndTime'])
            curr_time = pd.to_timedelta(scene_data.loc[i, 'StartTime'])
            time_diff = (curr_time - prev_time).total_seconds()

            # Normalized time (assuming max time diff is 10 minutes)
            max_time_diff = 10 * 60  # 10 minutes in seconds
            T0 = min(time_diff / max_time_diff, 1)  # Ensure T0 is in [0, 1]

            # Calculate influence and store in the corresponding row
            if prev_valency == 0:
                I0 = (transitions[0, 0] / (transitions[0, 0] + transitions[0, 1])) * (1 - T0)
                influences_0.append(I0)
                data.loc[scene_data.index[i - 1], 'Influence_0'] = I0
            else:
                I1 = (transitions[1, 0] / (transitions[1, 0] + transitions[1, 1])) * (1 - T0)
                influences_1.append(I1)
                data.loc[scene_data.index[i - 1], 'Influence_1'] = I1

        # Update sequence length
        if current_emotion == curr_valency:
            sequence_length += 1
        else:
            sequence_length = 1
            current_emotion = curr_valency

        data.loc[scene_data.index[i], 'Sequence_Length'] = sequence_length

# Calculate transition probabilities
total_transitions = transitions.sum(axis=1, keepdims=True)
transition_matrix = transitions / total_transitions

# Calculate average influence values
average_influence_0 = np.mean(influences_0) if influences_0 else 0
average_influence_1 = np.mean(influences_1) if influences_1 else 0

# Print results (for verification)
print("Transition Matrix:")
print(transition_matrix)

print("\nAverage Influence Values:")
print(f"I0: {average_influence_0}")
print(f"I1: {average_influence_1}")

print("\nEmotion Sequence Length:")
print(data[['Sr No.', 'scene_number', 'valency', 'Sequence_Length']])

# Save the updated data back to an Excel file
data.to_excel(file_path, index=False)
