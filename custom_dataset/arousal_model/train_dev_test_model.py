import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import os
import pickle

# Directory settings
train_directory = '../Train'
dev_directory = '../Dev'
test_directory = '../Test'
models_directory = 'Models'
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

# Function to evaluate model performance
def evaluate_model(X, y, model):
    y_pred = model.predict(X)
    accuracy = accuracy_score(y, y_pred)
    precision = precision_score(y, y_pred)
    recall = recall_score(y, y_pred)
    f1 = f1_score(y, y_pred)
    roc_auc = roc_auc_score(y, y_pred)
    return accuracy, precision, recall, f1, roc_auc

# Train, Dev, and Test implementation for each speaker
def train_dev_test_for_speaker(speaker, train_file_path, dev_file_path, test_file_path, feature_columns, target_column):
    # Load train, dev, and test datasets
    df_train = pd.read_csv(train_file_path)
    df_dev = pd.read_csv(dev_file_path)
    df_test = pd.read_csv(test_file_path)

    # Prepare training data
    X_train, y_train, train_scaler = prepare_data(df_train, feature_columns, target_column)
    X_dev, y_dev, _ = prepare_data(df_dev, feature_columns, target_column)
    X_test, y_test, _ = prepare_data(df_test, feature_columns, target_column)

    # Initialize model
    model = RandomForestClassifier(n_estimators=100, random_state=42)

    # Train the initial model
    model.fit(X_train, y_train)

    # Iteratively improve the model using the dev dataset
    X_train_extended = X_train.tolist()
    y_train_extended = y_train.tolist()

    for i in range(X_dev.shape[0]):
        X_current = X_dev[i].reshape(1, -1)
        y_pred = model.predict(X_current)[0]

        # Add the predicted value to the training set
        X_train_extended.append(X_dev[i].tolist())
        y_train_extended.append(y_pred)

        # Retrain the model with the extended training set
        model.fit(X_train_extended, y_train_extended)

    # Save the improved model
    model_path = os.path.join(models_directory, f'{speaker}_model.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"Improved model for {speaker} saved to {model_path}")

    # Evaluate the improved model on the test dataset
    accuracy_test, precision_test, recall_test, f1_test, roc_auc_test = evaluate_model(X_test, y_test, model)

    print(f"\nTest set - Model performance for {speaker}:")
    # Print number of samples in Train, Dev, and Test datasets
    print(f"Number of samples in Train dataset: {len(y_train)}")
    print(f"Number of samples in Dev dataset: {len(y_dev)}")
    print(f"Number of samples in Test dataset: {len(y_test)}")
    print(f"Accuracy: {accuracy_test}")
    print(f"Precision: {precision_test}")
    print(f"Recall: {recall_test}")
    print(f"F1-score: {f1_test}")
    print(f"ROC AUC: {roc_auc_test}")

# Define feature columns and target column
feature_columns = ['Influence_0', 'Influence_1', 'ElapsedTime_0', 'ElapsedTime_1', 'Sequence_Length']
target_column = 'valence'

# Process all speakers present in the directories
train_files = os.listdir(train_directory)
dev_files = os.listdir(dev_directory)
test_files = os.listdir(test_directory)

# Ensure that only common files between train, dev, and test directories are processed
common_files = set(train_files).intersection(dev_files).intersection(test_files)

for file_name in common_files:
    speaker = os.path.splitext(file_name)[0]
    train_file_path = os.path.join(train_directory, file_name)
    dev_file_path = os.path.join(dev_directory, file_name)
    test_file_path = os.path.join(test_directory, file_name)
    train_dev_test_for_speaker(speaker, train_file_path, dev_file_path, test_file_path, feature_columns, target_column)
