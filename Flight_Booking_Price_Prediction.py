# Importing the Libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Loading the Data
df = pd.read_csv("C:\\Python_Scripts\\Flight_Booking.csv")
df = df.drop(columns=['Unnamed: 0'])  # Fixed the quotes issue

# Data Exploration
print(df.head())
print(df.shape)
print(df.info())
print(df.describe())
print(df.isnull().sum())

# Convert categorical columns to numeric using one-hot encoding
categorical_cols = ['airline', 'source_city', 'destination_city', 'class', 'departure_time']
df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

# Data Visualization
plt.figure(figsize=(15, 5))
sns.lineplot(x=df['airline'], y=df['price'])
plt.title('Airlines Vs Price', fontsize=15)
plt.xlabel('Airline', fontsize=15)
plt.ylabel('Price', fontsize=15)
plt.xticks(rotation=45)  # Rotate x-tick labels for better readability
plt.show()

plt.figure(figsize=(15, 5))
sns.lineplot(data=df, x='days_left', y='price', color='blue')
plt.title('Days Left for Departure Versus Ticket Price', fontsize=15)
plt.xlabel('Days Left for Departure', fontsize=15)
plt.ylabel('Price', fontsize=15)
plt.show()

plt.figure(figsize=(10, 5))
sns.barplot(x='airline', y='price', data=df)
plt.title('Average Price per Airline', fontsize=15)  # Added title for clarity
plt.show()

plt.figure(figsize=(10, 8))
sns.barplot(x='class', y='price', data=df, hue='airline')
plt.title('Price by Class and Airline', fontsize=15)  # Added title for clarity
plt.show()

# Subplot for Days Left vs Price by Cities
fig, ax = plt.subplots(1, 2, figsize=(20, 6))  # Corrected subplot creation
sns.lineplot(x='days_left', y='price', data=df, hue='source_city', ax=ax[0])
ax[0].set_title('Price vs Days Left by Source City')  # Added title for clarity
sns.lineplot(x='days_left', y='price', data=df, hue='destination_city', ax=ax[1])
ax[1].set_title('Price vs Days Left by Destination City')  # Added title for clarity
plt.show()

# Feature Selection - Correlation Matrix
# Ensure that only numeric columns are included in correlation calculation
numeric_df = df_encoded.select_dtypes(include=[np.number])

plt.figure(figsize=(10, 5))
sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm")
plt.title('Correlation Heatmap', fontsize=15)  # Added title for clarity
plt.show()

# Additional Visualizations
plt.figure(figsize=(15, 10))
plt.subplot(4, 2, 1)
sns.countplot(x=df['source_city'], data=df)
plt.title("Frequency of Source City")

plt.subplot(4, 2, 2)
sns.countplot(x=df['departure_time'], data=df)
plt.title("Frequency of Departure Time")

plt.tight_layout()  # Adjust subplots to fit in figure area
plt.show()
