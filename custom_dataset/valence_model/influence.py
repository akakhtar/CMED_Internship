import pandas as pd
import numpy as np
import os

directory = "Dev"


def normalize_valence(val):
    return 0 if val == 0 else 1


def transition_matrix(data):
    transition_counts = np.zeros((2, 2))
    valence_transition = []
    for i in range(1, len(data)):
        prev_valence = normalize_valence(data.loc[i - 1, 'valence'])
        curr_valence = normalize_valence(data.loc[i, 'valence'])
        valence_transition.append([prev_valence, curr_valence])

    valence_transition = np.array(valence_transition)
    for transition in valence_transition:
        prev_valence, curr_valence = transition
        transition_counts[prev_valence, curr_valence] += 1

    total_transition = transition_counts.sum(axis=1)
    with np.errstate(divide='ignore', invalid='ignore'):
        transition_probabilities = np.divide(transition_counts, total_transition[:, None],
                                             where=total_transition[:, None] != 0)
        transition_probabilities[total_transition == 0] = 0

    return transition_probabilities


def calculate_influence(data, transition_probabilities):
    for scene in data['Scene_ID'].unique():
        scene_data = data[data['Scene_ID'] == scene].reset_index()
        curr_emotion = None
        sequence_length = 1
        last_time_0 = None
        last_time_1 = None

        for i in range(len(scene_data)):
            curr_index = scene_data.loc[i, 'index']
            curr_time = scene_data.loc[i, 'StartTime']
            curr_valence = normalize_valence(scene_data.loc[i, 'valence'])
            influence_0 = 0
            influence_1 = 0

            if i == 0:
                elapsed_time = [1, 1]
            else:
                elapsed_time = [float("inf"), float('inf')]
                if last_time_0 is not None:
                    elapsed_time[0] = (pd.to_datetime(curr_time) - pd.to_datetime(last_time_0)).total_seconds()
                if last_time_1 is not None:
                    elapsed_time[1] = (pd.to_datetime(curr_time) - pd.to_datetime(last_time_1)).total_seconds()

                max_time = max(elapsed_time)

                if max_time == float('inf'):
                    elapsed_time = [1 if t == float('inf') else 0 for t in elapsed_time]
                elif max_time > 0:
                    elapsed_time = [t / max_time for t in elapsed_time]

                # data.loc[curr_index, 'ElapsedTime_0'] = elapsed_time[0]
                # data.loc[curr_index, 'ElapsedTime_1'] = elapsed_time[1]
                prev_valence = normalize_valence(scene_data.loc[i - 1, 'valence'])

                # Handle influence calculation based on previous valence
                if prev_valence == 0:
                    influence_0 = transition_probabilities[0, 0] * (1 - elapsed_time[0])
                    influence_1 = transition_probabilities[0, 1] * (1 - elapsed_time[1])
                elif prev_valence == 1:
                    influence_0 = transition_probabilities[1, 0] * (1 - elapsed_time[0])
                    influence_1 = transition_probabilities[1, 1] * (1 - elapsed_time[1])

            data.loc[curr_index, 'Influence_0'] = influence_0
            data.loc[curr_index, 'Influence_1'] = influence_1
            data.loc[curr_index, 'ElapsedTime_0'] = elapsed_time[0]
            data.loc[curr_index, 'ElapsedTime_1'] = elapsed_time[1]
            if curr_valence == 0:
                last_time_0 = curr_time
            else:
                last_time_1 = curr_time

            if curr_emotion == curr_valence:
                sequence_length += 1
            else:
                sequence_length = 1
                curr_emotion = curr_valence
            data.loc[curr_index,'Sequence_Length'] = sequence_length

    return data


for filename in os.listdir(directory):
    if filename.endswith('.csv'):
        file_path = os.path.join(directory, filename)
        df = pd.read_csv(file_path)
        df['Influence_0'] = float(0)
        df['Influence_1'] = float(0)
        df['ElapsedTime_0'] = float('inf')
        df['ElapsedTime_1'] = float('inf')
        df['Sequence_Length'] = 0

        transition_matrix_value = transition_matrix(df)
        print(f"Transition matrix for {filename} :\n{transition_matrix_value}\n")
        df = calculate_influence(df, transition_matrix_value)
        df.to_csv(file_path, index=False)
