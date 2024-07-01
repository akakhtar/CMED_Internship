<<<<<<< HEAD
import pandas as pd
import numpy as np

# Load the dataset
file_path = 'Speakers_Valency/Ross_v.xlsx'  # Change to the actual file path
data = pd.read_excel(file_path)

# Function to normalize valency to 0 or 1
def normalize_valency(val):
    return 0 if val == 0 else 1

# Initialize a DataFrame to store transition probabilities for each scene
transition_prob_df = pd.DataFrame(columns=['scene_number', 'p00', 'p01', 'p10', 'p11'])

# Maximum time difference (e.g., 10 minutes in seconds)
T_max = 10 * 60

# Calculate transition probabilities for each scene
for scene in data['scene_number'].unique():
    scene_data = data[data['scene_number'] == scene].reset_index()

    # Initialize transition counts for the current scene
    transition_counts = np.zeros((2, 2))  # 2x2 matrix for valency 0 and 1

    # Collect transitions for the current scene
    valency_transitions = []
    for i in range(1, len(scene_data)):
        prev_valency = normalize_valency(scene_data.loc[i - 1, 'valency'])
        curr_valency = normalize_valency(scene_data.loc[i, 'valency'])
        valency_transitions.append([prev_valency, curr_valency])

    # Convert valency transitions to numpy array
    valency_transitions = np.array(valency_transitions)

    # Calculate transition counts
    for transition in valency_transitions:
        prev_valency, curr_valency = transition
        transition_counts[prev_valency, curr_valency] += 1

    # Calculate total transitions from each state (n_i)
    total_transitions = transition_counts.sum(axis=1)

    # Calculate transition probabilities
    with np.errstate(divide='ignore', invalid='ignore'):
        transition_probabilities = np.divide(transition_counts, total_transitions[:, None], where=total_transitions[:, None] != 0)
        transition_probabilities[total_transitions == 0] = 0  # Set probabilities to 0 where total transitions are 0

    # Append transition probabilities to the DataFrame
    transition_prob_df = pd.concat([transition_prob_df, pd.DataFrame([{
        'scene_number': scene,
        'p00': transition_probabilities[0, 0],
        'p01': transition_probabilities[0, 1],
        'p10': transition_probabilities[1, 0],
        'p11': transition_probabilities[1, 1]
    }])], ignore_index=True)

# Print transition probabilities DataFrame
print(transition_prob_df)

# Add columns for influence values
data['Influence_0'] = 0
data['Influence_1'] = 0

# Calculate influence values for each row
for scene in data['scene_number'].unique():
    scene_data = data[data['scene_number'] == scene].reset_index()
    scene_transitions = transition_prob_df[transition_prob_df['scene_number'] == scene].iloc[0]

    for i in range(1, len(scene_data)):
        prev_valency = normalize_valency(scene_data.loc[i - 1, 'valency'])
        curr_valency = normalize_valency(scene_data.loc[i, 'valency'])

        # Calculate the time difference between current and previous utterances
        prev_time = pd.to_timedelta(scene_data.loc[i - 1, 'EndTime']).total_seconds()
        curr_time = pd.to_timedelta(scene_data.loc[i, 'StartTime']).total_seconds()
        time_diff = curr_time - prev_time
        tau = min(time_diff / T_max, 1)  # Ensure tau is in [0, 1]

        # Calculate influence based on transition probabilities and tau
        if prev_valency == 0 and curr_valency == 0:
            influence = scene_transitions['p00'] * (1 - tau)
            data.loc[scene_data.loc[i, 'index'], 'Influence_0'] = influence
        elif prev_valency == 0 and curr_valency == 1:
            influence = scene_transitions['p01'] * (1 - tau)
            data.loc[scene_data.loc[i, 'index'], 'Influence_0'] = influence
        elif prev_valency == 1 and curr_valency == 0:
            influence = scene_transitions['p10'] * (1 - tau)
            data.loc[scene_data.loc[i, 'index'], 'Influence_1'] = influence
        elif prev_valency == 1 and curr_valency == 1:
            influence = scene_transitions['p11'] * (1 - tau)
            data.loc[scene_data.loc[i, 'index'], 'Influence_1'] = influence

# Calculate Emotion Sequence Length for each scene
data['Sequence_Length'] = 1

# Iterate through each scene
for scene in data['scene_number'].unique():
    scene_data = data[data['scene_number'] == scene].reset_index()
    sequence_length = 1

    for i in range(1, len(scene_data)):
        current_index = scene_data.loc[i, 'index']
        prev_index = scene_data.loc[i - 1, 'index']

        # Check if the current and previous valencies are the same
        if normalize_valency(scene_data.loc[i, 'valency']) == normalize_valency(scene_data.loc[i - 1, 'valency']):
            sequence_length += 1
        else:
            sequence_length = 1  # Reset the sequence length for a new emotion

        data.loc[current_index, 'Sequence_Length'] = sequence_length

# Print the updated DataFrame to verify
print(data[['scene_number', 'valency', 'Sequence_Length']])

# Save the updated data back to an Excel file
data.to_excel(file_path, index=False)
=======
>>>>>>> 137ac92d8917ed48151c21a2665631e9c9a60071
