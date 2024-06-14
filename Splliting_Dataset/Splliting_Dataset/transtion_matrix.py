import pandas as pd
import numpy as np

# Load the dataset
file_path = 'Speakers_Dataset/Rachel.xlsx'  # Change to the actual file path
data = pd.read_excel(file_path)

# Function to normalize valency to 0 or 1
def normalize_valency(val):
    return 0 if val == 0 else 1

# Iterate through each scene
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
    transition_probabilities = transition_counts / total_transitions[:, None]

    # Print transition counts (n_ij) and transition probabilities (p_ij) for the current scene
    print(f"\nScene {scene} Transition Counts (n_ij):")
    print(transition_counts)

    print(f"\nScene {scene} Transition Probabilities (p_ij):")
    print(transition_probabilities)

    # Print each n_ij, p_ij value, and n_i for the current scene
    print(f"\nScene {scene} Individual n_ij, p_ij values, and n_i:")
    for i in range(2):
        for j in range(2):
            print(f"n_{i}{j}: {transition_counts[i, j]}, p_{i}{j}: {transition_probabilities[i, j]}")
        print(f"n_{i}: {total_transitions[i]}")
