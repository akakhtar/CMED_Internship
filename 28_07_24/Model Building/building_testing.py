import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

train = pd.read_csv('../Self Report Features/Final Speakers Data/rachel_train.csv')
test = pd.read_csv('../Self Report Features/Final Speakers Data/rachel_test.csv')

print(f"Shape of Train Data Set : {train.shape}")
print(f"Shape of Test Data Set : {test.shape}")
print("\nCount of valence :")
print(f"Train: \n{train['valence'].value_counts()}")
print(f"Test : \n{test['valence'].value_counts()}")
print("\nCount of arousal :")
print(f"Train: \n{train['arousal'].value_counts()}")
print(f"Test : \n{test['arousal'].value_counts()}")

X_train = train[['Influence_0', 'Influence_1', 'Sequence_Length','audio_feature','facial_feature']]
y_train = train['valence']

X_test = test[['Influence_0', 'Influence_1', 'Sequence_Length','audio_feature','facial_feature']]
y_test = test['valence']

model = RandomForestClassifier(n_estimators=100, criterion='entropy', max_depth=10, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(f"Classification Report :\n{classification_report(y_test, y_pred)}")

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Predicted Negative', 'Predicted Positive'],
            yticklabels=['Actual Negative', 'Actual Positive'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()
