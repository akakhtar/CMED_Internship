import pandas as pd

# Load the dataset
file_path = 'Speakers_Valency/Ross_test.xlsx'  # Change to the actual file path
data = pd.read_excel(file_path)

# Sort the data by dialogue_id and utterance_number in ascending order
data_sorted = data.sort_values(by=['Dialogue_ID', 'Utterance_ID'], ascending=[True, True])

# Reset index after sorting
data_sorted = data_sorted.reset_index(drop=True)

# Save the sorted data back to an Excel file
data_sorted.to_excel(file_path, index=False)

# Print the sorted DataFrame to verify
#print(data_sorted[['dialogue_id', 'utterance_number']])
