import pandas as pd
import os
directory = 'Dev'
def sort_file(file_path):
    df = pd.read_csv(file_path)
    df_sorted = df.sort_values(by=['Dialogue_ID','Utterance_ID'],ascending=[True,True])
    df_sorted = df_sorted.reset_index(drop=True)
    df_sorted.to_csv(file_path,index = False)

for filename in os.listdir(directory):
    if filename.endswith('.csv'):
        file_path = os.path.join(directory, filename)
        sort_file(file_path)

print("Sorting complete.")
