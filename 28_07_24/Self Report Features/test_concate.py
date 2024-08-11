import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

df_train = pd.read_csv("../Speakers/Train/Rachel_train.csv")
df_dev = pd.read_csv("../Speakers/Dev/Rachel_dev.csv")
df_test = pd.read_csv("../Speakers/Test/Rachel_test.csv")

df = pd.concat([df_train,df_dev,df_test],ignore_index=False)
print(f"Dialogue id in joint df:\n{df['Dialogue_ID']}")

df = df.sort_values(by=['Dialogue_ID','Utterance_ID'],ascending=[True,True])
df = df.reset_index(drop=True)
df = df.drop(columns=['Scene_ID','Influence_0','Influence_1','Sequence_Length'])

# Print the entire sorted DataFrame
print("Sorted DataFrame:")
print(df.shape)
count = df["landmarks"].value_counts().get("{}", 0)
print(f"No of data points with no face matched or found : {count}")
df = df[(df["landmarks"] != '{}') & (df["landmarks"] != 0)]
print(f"Shape of Combined Data Set after removing the unmatched or not found faces : {df.shape}")
print(df.shape)

def add_scene_number():
    df["Scene_ID"] = 0
    current_scene_number = 1
    prev_dial_id = df.loc[0, 'Dialogue_ID']

    for index, row in df.iterrows():
        current_dial_id = row['Dialogue_ID']
        if current_dial_id != prev_dial_id:
            current_scene_number += 1
            prev_dial_id = current_dial_id
        df.at[index, 'Scene_ID'] = current_scene_number

add_scene_number()
df.to_csv('Final Speakers Data/rachel_joint.csv',index=False)

# train, test = train_test_split(df, test_size=0.25, shuffle=False)
# print(f"Shape of Train Data Set : {train.shape}")
# print(f"Shape of Test Data Set : {test.shape}")