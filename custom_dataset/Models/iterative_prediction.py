import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import os
import pickle

# Directory and file settings
directory = "Train"
models_directory = "Models"

if not os.path.exists(models_directory):
    os.makedirs(models_directory)

# Function to prepare the data
def prepare_data(data, feature_columns, target_column):
    data = data.dropna()

    X = data[feature_columns]
    y = data[target_column]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y, scaler

# Function to train and predict iteratively
def train_and_predict_iteratively(X_train, y_train, X_test, model):
    model.fit(X_train, y_train)

    predictions = []
    X_train_extended = X_train.tolist()
    y_train_extended = y_train.tolist()

    for i in range(X_test.shape[0]):
        X_current = X_test[i].reshape(1, -1)
        y_pred = model.predict(X_current)[0]
        predictions.append(y_pred)

        # Add the predicted value to the training set
        X_train_extended.append(X_test[i].tolist())
        y_train_extended.append(y_pred)

        # Retrain the model with the extended training set
        model.fit(X_train_extended, y_train_extended)

    return np.array(predictions)

# Phase 1 and Phase 2 implementation
def phase1_phase2(file_path, feature_columns, target_column):
    df = pd.read_csv(file_path)

    # Phase 1: First 60% training and 20% testing
    split_1 = int(0.6 * len(df))
    split_2 = int(0.8 * len(df))
    split_3 = int(0.2 * len(df))

    X_scaled, y, scaler = prepare_data(df, feature_columns, target_column)

    X_train_phase1 = X_scaled[:split_1]
    y_train_phase1 = y[:split_1]
    X_test_phase1 = X_scaled[split_1:split_2]
    y_test_phase1 = y[split_1:split_2]

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    y_pred_phase1 = train_and_predict_iteratively(X_train_phase1, y_train_phase1, X_test_phase1, model)

    accuracy_phase1 = accuracy_score(y_test_phase1, y_pred_phase1)
    precision_phase1 = precision_score(y_test_phase1, y_pred_phase1)
    recall_phase1 = recall_score(y_test_phase1, y_pred_phase1)
    f1_phase1 = f1_score(y_test_phase1, y_pred_phase1)
    roc_auc_phase1 = roc_auc_score(y_test_phase1, y_pred_phase1)

    print(f"Phase 1 - Model performance:")
    # Print number of samples in Train and Test datasets
    print(f"Number of samples in Train dataset: {len(X_train_phase1)}")
    print(f"Number of samples in Test dataset: {len(X_test_phase1)}")
    print(f"Accuracy: {accuracy_phase1}")
    print(f"Precision: {precision_phase1}")
    print(f"Recall: {recall_phase1}")
    print(f"F1-score: {f1_phase1}")
    print(f"ROC AUC: {roc_auc_phase1}")

    # # Print predicted and original values for Phase 1
    # print("\nPhase 1 - Predicted vs Original values:")
    # for i in range(len(y_test_phase1)):
    #     print(f"Original: {y_test_phase1.iloc[i]}, Predicted: {y_pred_phase1[i]}")

    # Phase 2: Next 60% training and next 20% testing (ignoring the first 20%)
    X_train_phase2 = X_scaled[split_3:split_2]
    y_train_phase2 = y[split_3:split_2]
    X_test_phase2 = X_scaled[split_2:]
    y_test_phase2 = y[split_2:]

    y_pred_phase2 = train_and_predict_iteratively(X_train_phase2, y_train_phase2, X_test_phase2, model)

    accuracy_phase2 = accuracy_score(y_test_phase2, y_pred_phase2)
    precision_phase2 = precision_score(y_test_phase2, y_pred_phase2)
    recall_phase2 = recall_score(y_test_phase2, y_pred_phase2)
    f1_phase2 = f1_score(y_test_phase2, y_pred_phase2)
    roc_auc_phase2 = roc_auc_score(y_test_phase2, y_pred_phase2)

    print(f"Phase 2 - Model performance:")
    # Print number of samples in Train and Test datasets
    print(f"Number of samples in Train dataset: {len(X_train_phase2)}")
    print(f"Number of samples in Test dataset: {len(X_test_phase2)}")
    print(f"Accuracy: {accuracy_phase2}")
    print(f"Precision: {precision_phase2}")
    print(f"Recall: {recall_phase2}")
    print(f"F1-score: {f1_phase2}")
    print(f"ROC AUC: {roc_auc_phase2}")

    # Print predicted and original values for Phase 2
    # print("\nPhase 2 - Predicted vs Original values:")
    # for i in range(len(y_test_phase2)):
    #     print(f"Original: {y_test_phase2.iloc[i]}, Predicted: {y_pred_phase2[i]}")

# Define feature columns and target column
feature_columns = ['Influence_0', 'Influence_1', 'ElapsedTime_0', 'ElapsedTime_1', 'Sequence_Length']
target_column = 'valence'

# Apply the implementation to the file
file_path = '../../Influenece/test_Ross.csv'
phase1_phase2(file_path, feature_columns, target_column)
