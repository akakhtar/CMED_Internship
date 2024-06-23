import pandas as pd
import numpy as np

# Load the dataset
file_path = 'Ross_test.xlsx'  # Change to the actual file path
data = pd.read_excel(file_path)

# Function to normalize valency to 0 or 1
def normalize_valency(val):
    return 0 if val == 0 else 1

# Initialize columns for influence values, elapsed time, and previous valency
data['Influence_0'] = 0
data['Influence_1'] = 0
data['Elapsed_Time_0'] = 0
data['Elapsed_Time_1'] = 0
data['Previous_Valency'] = np.nan

# Compute transition counts and probabilities (already provided in previous steps)
transition_counts = np.zeros((2, 2))
valency_transitions = []
for i in range(1, len(data)):
    prev_valency = normalize_valency(data.loc[i - 1, 'valency'])
    curr_valency = normalize_valency(data.loc[i, 'valency'])
    valency_transitions.append([prev_valency, curr_valency])

valency_transitions = np.array(valency_transitions)
for transition in valency_transitions:
    prev_valency, curr_valency = transition
    transition_counts[prev_valency, curr_valency] += 1

total_transitions = transition_counts.sum(axis=1)
with np.errstate(divide='ignore', invalid='ignore'):
    transition_probabilities = np.divide(transition_counts, total_transitions[:, None], where=total_transitions[:, None] != 0)
    transition_probabilities[total_transitions == 0] = 0

# Ensure the data is sorted by 'Dialogue_ID' and 'Utterance_ID'
data = data.sort_values(by=['Dialogue_ID', 'Utterance_ID'])

print(f"Transition matrix : \n transition_probabilities}")

# Process each scene to compute influences and elapsed times
for scene in data['scene_number'].unique():
    scene_data = data[data['scene_number'] == scene].reset_index()
    last_time_0 = None
    last_time_1 = None

    for i in range(len(scene_data)):
        print(f"Dialogue_ID: {scene_data.loc[i, 'Dialogue_ID']}, Utterance_ID: {scene_data.loc[i, 'Utterance_ID']}")
        current_index = scene_data.loc[i, 'index']
        current_time = scene_data.loc[i, 'StartTime']
        print(f"Current Time: {current_time}")
        current_valency = normalize_valency(scene_data.loc[i, 'valency'])

        if i == 0:
            # First row in the scene, no previous valency to reference
            data.loc[current_index, 'Influence_0'] = 0
            data.loc[current_index, 'Influence_1'] = 0
            data.loc[current_index, 'Elapsed_Time_0'] = 0
            data.loc[current_index, 'Elapsed_Time_1'] = 0
            data.loc[current_index, 'Previous_Valency'] = np.nan
        else:
            # Calculate elapsed time vector
            elapsed_time = [0, 0]
            if last_time_0 is not None:
                elapsed_time[0] = (pd.to_datetime(current_time) - pd.to_datetime(last_time_0)).total_seconds()
            if last_time_1 is not None:
                elapsed_time[1] = (pd.to_datetime(current_time) - pd.to_datetime(last_time_1)).total_seconds()
            print(elapsed_time)
            max_time = max(elapsed_time)
            print(f"Maximum Time: {max_time}")
            if max_time > 0:
                elapsed_time = [t / max_time for t in elapsed_time]
            print(f"Elapsed Time : {elapsed_time}")

            # Store elapsed time and previous valency in the DataFrame
            data.loc[current_index, 'Elapsed_Time_0'] = elapsed_time[0]
            data.loc[current_index, 'Elapsed_Time_1'] = elapsed_time[1]
            prev_valency = normalize_valency(scene_data.loc[i - 1, 'valency'])
            data.loc[current_index, 'Previous_Valency'] = prev_valency

            # Determine the previous valency to select appropriate transition probabilities
            if prev_valency == 0:
                influence_0 = transition_probabilities[0, 0] * (1 - elapsed_time[0])
                influence_1 = transition_probabilities[0, 1] * (1 - elapsed_time[1])
                data.loc[current_index, 'Influence_0'] = influence_0
                data.loc[current_index, 'Influence_1'] = influence_1
            elif prev_valency == 1:
                influence_0 = transition_probabilities[1, 0] * (1 - elapsed_time[0])
                influence_1 = transition_probabilities[1, 1] * (1 - elapsed_time[1])
                data.loc[current_index, 'Influence_0'] = influence_0
                data.loc[current_index, 'Influence_1'] = influence_1

        # Update last occurrence times
        if current_valency == 0:
            last_time_0 = current_time
        else:
            last_time_1 = current_time

# Print the updated DataFrame to verify
print(data[['scene_number', 'valency', 'Previous_Valency', 'Influence_0', 'Influence_1', 'Elapsed_Time_0', 'Elapsed_Time_1']])

# Save the updated data back to an Excel file
data.to_excel('Updated_' + file_path, index=False)
