# customer_churn_analysis.py
# Python - Project-3-Customer Churn Dataset COMPLETED by Nagesha KS

# Import Required Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, confusion_matrix, accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# Step 1: Load the Dataset
df = pd.read_csv('C:/Python_Scripts/customer_churn.csv')

# Step 2: Data Preparation
# Display the first few rows for an overview
print(df.head())

# Check for missing values
print(df.isnull().sum())

# Fill missing values if needed (this is a general approach)
# df.fillna(method='ffill', inplace=True)
df.ffill(inplace=True)


# Convert categorical columns to appropriate types for analysis
categorical_columns = ['gender', 'Partner', 'Dependents', 'PhoneService', 
                       'MultipleLines', 'InternetService', 'Contract', 'PaymentMethod', 'Churn']
df[categorical_columns] = df[categorical_columns].apply(lambda x: x.astype('category'))

# Step 3: Data Manipulation
# Extract the 5th column and store it in 'customer_5'
customer_5 = df.iloc[:, 4]

# Extract the 15th column and store it in 'customer_15'
customer_15 = df.iloc[:, 14]

# Extract all male senior citizens whose payment method is electronic check
senior_male_electronic = df[(df['gender'] == 'Male') & 
                            (df['SeniorCitizen'] == 1) & 
                            (df['PaymentMethod'] == 'Electronic check')]

# Extract customers with tenure > 70 months or monthly charges > 100
customer_total_tenure = df[(df['tenure'] > 70) | (df['MonthlyCharges'] > 100)]

# Extract customers with a two-year contract, payment method as mailed check, and churned
two_mail_yes = df[(df['Contract'] == 'Two year') & 
                  (df['PaymentMethod'] == 'Mailed check') & 
                  (df['Churn'] == 'Yes')]

# Extract 333 random records
customer_333 = df.sample(n=333, random_state=42)

# Count different levels in 'Churn' column
churn_counts = df['Churn'].value_counts()
print(churn_counts)

# Step 4: Data Visualization
# Bar Plot for Internet Service Categories
plt.figure(figsize=(8, 5))
sns.countplot(x='InternetService', data=df, color='orange')
plt.xlabel('Categories of Internet Service')
plt.ylabel('Count of Categories')
plt.title('Distribution of Internet Service')
plt.show()

# Histogram for Tenure
plt.figure(figsize=(8, 5))
plt.hist(df['tenure'], bins=30, color='green', edgecolor='black')
plt.title('Distribution of Tenure')
plt.xlabel('Tenure')
plt.ylabel('Frequency')
plt.show()

# Scatter Plot Between Monthly Charges and Tenure
plt.figure(figsize=(8, 5))
plt.scatter(df['tenure'], df['MonthlyCharges'], color='brown', alpha=0.6)
plt.xlabel('Tenure of customer')
plt.ylabel('Monthly Charges of customer')
plt.title('Tenure vs Monthly Charges')
plt.show()

# Box Plot Between Tenure and Contract
plt.figure(figsize=(8, 5))
sns.boxplot(x='Contract', y='tenure', data=df)
plt.title('Tenure by Contract Type')
plt.xlabel('Contract Type')
plt.ylabel('Tenure')
plt.show()

# Step 5: Linear Regression
# Independent and dependent variables
X = df[['tenure']]
y = df['MonthlyCharges']

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train the linear regression model
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)

# Predict on the test set
y_pred = linear_model.predict(X_test)

# Calculate RMSE
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f'Root Mean Square Error (Linear Regression): {rmse}')

# Store error in 'error'
error = y_test - y_pred

# Step 6: Logistic Regression
# Encode 'Churn' as binary
df['Churn'] = df['Churn'].apply(lambda x: 1 if x == 'Yes' else 0)

X = df[['MonthlyCharges']]
y = df['Churn']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.35, random_state=42)

# Train logistic regression model
logistic_model = LogisticRegression()
logistic_model.fit(X_train, y_train)

# Predict and evaluate
y_pred = logistic_model.predict(X_test)
conf_matrix = confusion_matrix(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
print(f'Confusion Matrix (Logistic Regression):\n{conf_matrix}')
print(f'Accuracy (Logistic Regression): {accuracy}')

# Multiple Logistic Regression
X = df[['tenure', 'MonthlyCharges']]
y = df['Churn']

# Split data into training and testing sets (80:20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
logistic_model_multi = LogisticRegression()
logistic_model_multi.fit(X_train, y_train)

# Predict and evaluate
y_pred_multi = logistic_model_multi.predict(X_test)
conf_matrix_multi = confusion_matrix(y_test, y_pred_multi)
accuracy_multi = accuracy_score(y_test, y_pred_multi)
print(f'Confusion Matrix (Multiple Logistic Regression):\n{conf_matrix_multi}')
print(f'Accuracy (Multiple Logistic Regression): {accuracy_multi}')

# Step 7: Decision Tree
X = df[['tenure']]
y = df['Churn']

# Split into training and testing sets (80:20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the decision tree model
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)

# Predict and evaluate
y_pred_dt = dt_model.predict(X_test)
conf_matrix_dt = confusion_matrix(y_test, y_pred_dt)
accuracy_dt = accuracy_score(y_test, y_pred_dt)
print(f'Confusion Matrix (Decision Tree):\n{conf_matrix_dt}')
print(f'Accuracy (Decision Tree): {accuracy_dt}')

# Step 8: Random Forest
X = df[['tenure', 'MonthlyCharges']]
y = df['Churn']

# Split data into training and testing sets (70:30)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train the random forest model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Predict and evaluate
y_pred_rf = rf_model.predict(X_test)
conf_matrix_rf = confusion_matrix(y_test, y_pred_rf)
accuracy_rf = accuracy_score(y_test, y_pred_rf)
print(f'Confusion Matrix (Random Forest):\n{conf_matrix_rf}')
print(f'Accuracy (Random Forest): {accuracy_rf}')

# Run the script:       python customer_churn_analysis.py
