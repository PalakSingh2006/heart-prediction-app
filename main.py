import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

# Settings
pd.options.display.float_format = '{:.2f}'.format
warnings.filterwarnings('ignore')

# Load dataset
data = pd.read_csv('heart.csv')

# Basic info
print(data.head())
print(data.shape)
print(data.columns)
print(data.info())

# Missing values heatmap
sns.heatmap(data.isnull(), cmap='magma', cbar=False)
plt.show()

# Description
print(data.describe().T)

# Separate target classes
yes = data[data['HeartDisease'] == 1].describe().T
no = data[data['HeartDisease'] == 0].describe().T

colors = ['#F93822', '#FDD20E']

# Heatmaps
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(5, 5))

plt.subplot(1, 2, 1)
sns.heatmap(yes[['mean']], annot=True, cmap=colors,
            linewidths=0.4, linecolor='black', cbar=False, fmt='.2f')
plt.title('Heart Disease')

plt.subplot(1, 2, 2)
sns.heatmap(no[['mean']], annot=True, cmap=colors,
            linewidths=0.4, linecolor='black', cbar=False, fmt='.2f')
plt.title('No Heart Disease')

fig.tight_layout(pad=2)
plt.show()

# Feature separation
col = list(data.columns)
categorical_features = []
numerical_features = []

for i in col:
    if len(data[i].unique()) > 6:
        numerical_features.append(i)
    else:
        categorical_features.append(i)

print('Categorical Features:', categorical_features)
print('Numerical Features:', numerical_features)

# Encoding
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

df1 = data.copy(deep=True)

df1['Sex'] = le.fit_transform(df1['Sex'])
df1['ChestPainType'] = le.fit_transform(df1['ChestPainType'])
df1['RestingECG'] = le.fit_transform(df1['RestingECG'])
df1['ExerciseAngina'] = le.fit_transform(df1['ExerciseAngina'])
df1['ST_Slope'] = le.fit_transform(df1['ST_Slope'])

# Distribution plots
fig, ax = plt.subplots(nrows=3, ncols=2, figsize=(10, 15))

for i in range(len(categorical_features) - 1):
    plt.subplot(3, 2, i + 1)
    sns.histplot(df1[categorical_features[i]], kde=True, color=colors[0])
    plt.title('Distribution: ' + categorical_features[i])

plt.tight_layout()
plt.show()

# ==============================
# ✅ MODEL TRAINING (ADDED PART)
# ==============================

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Features and target
X = df1.drop('HeartDisease', axis=1)
y = df1['HeartDisease']

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model
model = LogisticRegression(max_iter=1000)

# Train
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Accuracy
acc = accuracy_score(y_test, y_pred)
print("Model Accuracy:", acc)

# ==============================
# ✅ SAVE MODEL (IMPORTANT)
# ==============================

import joblib

joblib.dump(model, 'heart_model.pkl')

print("Model saved as heart_model.pkl")