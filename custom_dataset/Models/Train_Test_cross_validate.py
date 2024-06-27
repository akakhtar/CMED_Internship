import pandas as pd
import numpy as np
from sklearn.model_selection import cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import os
import pickle

# Directory and file settings
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


# Function to train and evaluate the model on separate training and testing datasets
def train_and_evaluate(train_file_path, test_file_path, feature_columns, target_column):
    # Load training data
    train_df = pd.read_csv(train_file_path)
    X_train_scaled, y_train, train_scaler = prepare_data(train_df, feature_columns, target_column)

    # Load testing data
    test_df = pd.read_csv(test_file_path)
    X_test_scaled, y_test, test_scaler = prepare_data(test_df, feature_columns, target_column)

    # Print number of samples in Train and Test datasets
    print(f"Number of samples in Train dataset: {len(train_df)}")
    print(f"Number of samples in Test dataset: {len(test_df)}")
    model = RandomForestClassifier(n_estimators=100, random_state=42)

    # Cross-validation on training data
    print(f"Cross-validation results on training data:")
    evaluate_model_with_cv(X_train_scaled, y_train, model)

    # Train the model on the entire training dataset
    model.fit(X_train_scaled, y_train)

    # Evaluate the model on the testing dataset
    y_pred_test = model.predict(X_test_scaled)

    accuracy_test = accuracy_score(y_test, y_pred_test)
    precision_test = precision_score(y_test, y_pred_test)
    recall_test = recall_score(y_test, y_pred_test)
    f1_test = f1_score(y_test, y_pred_test)
    roc_auc_test = roc_auc_score(y_test, y_pred_test)

    print(f"\nModel performance on testing data:")
    print(f"Accuracy: {accuracy_test}")
    print(f"Precision: {precision_test}")
    print(f"Recall: {recall_test}")
    print(f"F1-score: {f1_test}")
    print(f"ROC AUC: {roc_auc_test}")

    # Print predicted and original values for testing data
    print("\nTesting data - Predicted vs Original values:")
    for i in range(len(y_test)):
        print(f"Original: {y_test.iloc[i]}, Predicted: {y_pred_test[i]}")


# Define feature columns and target column
feature_columns = ['Influence_0', 'Influence_1', 'ElapsedTime_0', 'ElapsedTime_1', 'Sequence_Length']
target_column = 'valence'

# Apply the implementation to the files
train_file_path = '../Train/Ross.csv'
test_file_path = '../Test/Ross.csv'
train_and_evaluate(train_file_path, test_file_path, feature_columns, target_column)
