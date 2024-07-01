import pandas as pd
import os
directory = 'Dev'

def add_scene_number(file_path):
    df = pd.read_csv(file_path)
    df["Scene_ID"] = 0
    current_scene_number = 1
    prev_dial_id = df.loc[0, 'Dialogue_ID']

    for index, row in df.iterrows():
        current_dial_id = row['Dialogue_ID']
        if current_dial_id != prev_dial_id:
            current_scene_number += 1
            prev_dial_id = current_dial_id
        df.at[index, 'Scene_ID'] = current_scene_number
    df.to_csv(file_path, index=False)


for filename in os.listdir(directory):
    if filename.endswith('.csv'):
        file_path = os.path.join(directory, filename)
        add_scene_number(file_path)

print("Adding of the scene number done.")
