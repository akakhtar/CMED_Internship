#phase 1 60:20:  phase2  :60:20
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, make_scorer
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


# Function to evaluate model with cross-validation
def evaluate_model_with_cv(X, y, model):
    scoring = {
        'accuracy': 'accuracy',
        'precision': 'precision',
        'recall': 'recall',
        'f1': 'f1',
        'roc_auc': 'roc_auc'
    }

    cv_results = cross_validate(model, X, y, cv=5, scoring=scoring)

    for metric in scoring.keys():
        print(f"{metric}: {cv_results['test_' + metric].mean():.4f} (mean)")


# Phase 1 and Phase 2 implementation
def phase1_phase2(file_path, feature_columns, target_column):
    df = pd.read_csv(file_path)

    # Cross-validation evaluation
    X_scaled, y, scaler = prepare_data(df, feature_columns, target_column)
    model = RandomForestClassifier(n_estimators=100, random_state=42)

    print(f"Cross-validation results:")
    evaluate_model_with_cv(X_scaled, y, model)

    # Phase 1: First 60% training and 20% testing
    split_1 = int(0.6 * len(df))
    split_2 = int(0.8 * len(df))

    X_train_phase1 = X_scaled[:split_1]
    y_train_phase1 = y[:split_1]
    X_test_phase1 = X_scaled[split_1:split_2]
    y_test_phase1 = y[split_1:split_2]


    model.fit(X_train_phase1, y_train_phase1)
    y_pred_phase1 = model.predict(X_test_phase1)

    accuracy_phase1 = accuracy_score(y_test_phase1, y_pred_phase1)
    precision_phase1 = precision_score(y_test_phase1, y_pred_phase1)
    recall_phase1 = recall_score(y_test_phase1, y_pred_phase1)
    f1_phase1 = f1_score(y_test_phase1, y_pred_phase1)
    roc_auc_phase1 = roc_auc_score(y_test_phase1, y_pred_phase1)

    print(f"\nPhase 1 - Model performance:")
    # Print number of samples in Train and Test datasets
    print(f"Number of samples in Train dataset: {len(X_train_phase1)}")
    print(f"Number of samples in Test dataset: {len(X_test_phase1)}")
    print(f"Accuracy: {accuracy_phase1}")
    print(f"Precision: {precision_phase1}")
    print(f"Recall: {recall_phase1}")
    print(f"F1-score: {f1_phase1}")
    print(f"ROC AUC: {roc_auc_phase1}")

    # Print predicted and original values for Phase 1
    # print("\nPhase 1 - Predicted vs Original values:")
    # for i in range(len(y_test_phase1)):
    #     print(f"Original: {y_test_phase1.iloc[i]}, Predicted: {y_pred_phase1[i]}")

    # Phase 2: Next 60% training and next 20% testing (ignoring the first 20%)
    split_3 = int(0.2*len(df))
    X_train_phase2 = X_scaled[split_3:split_2]
    y_train_phase2 = y[split_3:split_2]
    X_test_phase2 = X_scaled[split_2:]
    y_test_phase2 = y[split_2:]



    model.fit(X_train_phase2, y_train_phase2)
    y_pred_phase2 = model.predict(X_test_phase2)

    accuracy_phase2 = accuracy_score(y_test_phase2, y_pred_phase2)
    precision_phase2 = precision_score(y_test_phase2, y_pred_phase2)
    recall_phase2 = recall_score(y_test_phase2, y_pred_phase2)
    f1_phase2 = f1_score(y_test_phase2, y_pred_phase2)
    roc_auc_phase2 = roc_auc_score(y_test_phase2, y_pred_phase2)

    print(f"\nPhase 2 - Model performance:")
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
file_path = '../Train/Ross.csv'
phase1_phase2(file_path, feature_columns, target_column)
