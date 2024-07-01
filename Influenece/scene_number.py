import pandas as pd

file_path = 'test_Ross.csv'
output_path = 'test_Ross.csv'

def add_scene_number(data):
    data["Scene_ID"] = 0
    curr_number = 1
    prev_episode = data.loc[0,'Episode']

    for index,row in data.iterrows():
        curr_episode = row['Episode']
        if curr_episode != prev_episode:
            curr_number += 1
            prev_episode = curr_episode
        data.at[index,'Scene_ID'] = curr_number
    data.to_csv(file_path,index = False)

df = pd.read_csv(file_path)

add_scene_number(df)









