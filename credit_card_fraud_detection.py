# credit_card_fraud_detection.py by Nagesha KS

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Load the dataset
data = pd.read_csv(r"C:\Python_Scripts\creditcard.csv")

#print("1-Column names after parsing:", data.columns)

# Check for the 'Class' column
if 'Class' in data.columns:
    print(data['Class'].value_counts())
else:
    print("2-Column 'Class' still not found.")

#print("3-Current working directory:", os.getcwd())
#print("4-Column names in the dataset:", data.columns)


# Exploratory Data Analysis (EDA)
print("Dataset Description:")
print(data.describe())

print("\nClass Distribution:")
print(data['Class'].value_counts())

# Visualize class distribution
data['Class'].value_counts().plot(kind='bar', title='Class Distribution')
plt.xlabel('Class')
plt.ylabel('Frequency')
plt.show()

# Preprocess the data
scaler = StandardScaler()
data['NormalizedAmount'] = scaler.fit_transform(data['Amount'].values.reshape(-1, 1))
data.drop(['Time', 'Amount'], axis=1, inplace=True)

import pandas as pd
from sklearn.preprocessing import StandardScaler

# Step 1: Load your data
#data = pd.read_csv('creditcard.csv')
data = pd.read_csv(r"C:\Python_Scripts\creditcard.csv")

# Step 2: Initial data inspection
print(data.head())  # Show the first few rows
print(data.dtypes)  # Check data types

# Step 3: Check for non-numeric values in 'Amount'
print(data['Amount'].unique())

# Step 4: Data preprocessing
# Remove rows with non-numeric 'Amount' entries (like '...')
data = data[data['Amount'].apply(lambda x: str(x).replace('.', '', 1).isdigit())]

# Convert 'Amount' column to numeric
data['Amount'] = pd.to_numeric(data['Amount'], errors='coerce')

# Drop rows with NaN values (if any were created during the conversion)
data.dropna(subset=['Amount'], inplace=True)

# Normalize the Amount column
scaler = StandardScaler()
data['NormalizedAmount'] = scaler.fit_transform(data['Amount'].values.reshape(-1, 1))

# Drop the original 'Amount' column if needed
data.drop(['Time', 'Amount'], axis=1, inplace=True)

# Step 5: Continue with your analysis or model training
print(data.head())

# Further steps could include splitting the data into training and testing sets, fitting models, etc.




# Split the data into training and testing sets
X = data.drop('Class', axis=1)
y = data['Class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Visualize confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()
