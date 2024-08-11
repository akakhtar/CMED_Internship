import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import KernelPCA

# train = pd.read_csv('../Speakers/Ross/ross_train.csv')
# test = pd.read_csv('../Speakers/Ross/ross_test.csv')
train = pd.read_csv('../Self Report Features/Final Speakers Data/rachel_train.csv')
test = pd.read_csv('../Self Report Features/Final Speakers Data/rachel_test.csv')

print(f"Spahe of Train Data Set : {train.shape}")
print(f"Spahe of Test Data Set : {test.shape}")


def flatten_audio_features(data, mfccs_col, mel_col, spectral_col):
    def str_to_list(s):
        return eval(s) if isinstance(s, str) else s

    data[mfccs_col] = data[mfccs_col].apply(str_to_list)
    data[mel_col] = data[mel_col].apply(str_to_list)
    data[spectral_col] = data[spectral_col].apply(str_to_list)

    flattened_data = []
    for i in range(len(data)):
        temp = []
        temp.extend(data[mfccs_col].iloc[i])
        temp.extend(data[mel_col].iloc[i])
        temp.extend(data[spectral_col].iloc[i])
        flattened_data.append(np.array(temp).flatten())
    return flattened_data


train['audio_combined_features'] = flatten_audio_features(train, 'mfccs', 'melSpectrogram', 'spectralContrast')
test['audio_combined_features'] = flatten_audio_features(test, 'mfccs', 'melSpectrogram', 'spectralContrast')

audio_feature_train = []
for index, row in train.iterrows():
    audio_feature_train.append(row['audio_combined_features'])

scaler = StandardScaler()
audio_scaled = scaler.fit_transform(audio_feature_train)

kpca = KernelPCA(n_components=1, kernel="rbf", eigen_solver='arpack')
audio_reduced = kpca.fit_transform(audio_scaled)

train["audio_feature"] = audio_reduced

audio_feature_test = []
for index, row in test.iterrows():
    audio_feature_test.append(row['audio_combined_features'])

audio_scaled_test = scaler.transform(audio_feature_test)
audio_reduced_test = kpca.transform(audio_scaled_test)
test["audio_feature"] = audio_reduced_test

# train.to_csv('../Speakers/Ross/ross_train.csv', index=False)
# print("New Train csv file for the speaker added with reduced audio feature!")
# test.to_csv('../Speakers/Ross/ross_test.csv', index=False)
# print("New Test csv file for the speaker added with reduced audio feature!")
train.to_csv('../Self Report Features/Final Speakers Data/rachel_train.csv', index=False)
test.to_csv('../Self Report Features/Final Speakers Data/rachel_test.csv', index=False)