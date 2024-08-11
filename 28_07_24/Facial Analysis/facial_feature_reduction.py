import pandas as pd
import numpy as np
import ast
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import KernelPCA

# df_train = pd.read_csv('../Speakers/Train/Ross_train.csv')
# df_test = pd.read_csv('../Speakers/Test/Ross_test.csv')
# df_dev = pd.read_csv('../Speakers/Dev/Rachel_dev.csv')
#
# df = pd.concat([df_train, df_test, df_dev], ignore_index=True)
df = pd.read_csv('../Self Report Features/Final Speakers Data/rachel_joint.csv')
print(f"Shape of Combined Data Set : {df.shape}")

count = df["landmarks"].value_counts().get("{}", 0)
print(f"No of data points with no face matched or found : {count}")
df = df[(df["landmarks"] != '{}') & (df["landmarks"] != 0) &(df["landmarks"].notna())]
print(f"Shape of Combined Data Set after removing the unmatched or not found faces : {df.shape}")
print(f"Columns in the dataset :\n{df.columns}")

train, test = train_test_split(df, test_size=0.2, shuffle=False)
print(f"Shape of Train Data Set : {train.shape}")
print(f"Shape of Test Data Set : {test.shape}")
print("\nCount of valence :")
print(f"Train: \n{train['valence'].value_counts()}")
print(f"Test : \n{test['valence'].value_counts()}")
print("\nCount of arousal :")
print(f"Train: \n{train['arousal'].value_counts()}")
print(f"Test : \n{test['arousal'].value_counts()}")

def extract_coordinates(landmarks_str):
    landmarks = ast.literal_eval(landmarks_str)
    coordinates = []
    for landmark in landmarks.values():
        coordinates.append(landmark['X'])
        coordinates.append(landmark['Y'])
    return np.array(coordinates).flatten()

train['landmarks'] = train['landmarks'].apply(extract_coordinates)
test['landmarks'] = test['landmarks'].apply(extract_coordinates)

landmarks_train = []
for index, row in train.iterrows():
    landmarks_train.append(row['landmarks'])

scaler = StandardScaler()
landmarks_scaled_train = scaler.fit_transform(landmarks_train)

kpca = KernelPCA(n_components=1, kernel="rbf", eigen_solver='arpack')
landmarks_reduced_train = kpca.fit_transform(landmarks_scaled_train)
train["facial_feature"] = landmarks_reduced_train
print(f"Shape of train dataset after adding facial_feature: {train.shape}")

landmarks_test = []
for index, row in test.iterrows():
    landmarks_test.append(row['landmarks'])

landmarks_scaled_test = scaler.transform(landmarks_test)
landmarks_reduced_test = kpca.transform(landmarks_scaled_test)
test["facial_feature"] = landmarks_reduced_test
print(f"Shape of test dataset after adding facial_feature: {test.shape}")

# train.to_csv('../Speakers/Ross/ross_train.csv', index=False)
# print("New Train csv file for the speaker added!")
# test.to_csv('../Speakers/Ross/ross_test.csv', index=False)
# print("New Test csv file for the speaker added!")
train.to_csv('../Self Report Features/Final Speakers Data/rachel_train.csv', index=False)
test.to_csv('../Self Report Features/Final Speakers Data/rachel_test.csv', index=False)
