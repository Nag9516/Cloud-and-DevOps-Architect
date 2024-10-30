# restaurant_revenue_prediction by Nagesha KS

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import joblib



df = pd.read_csv(r'C:\\Python_Scripts\\restaurant_revenue.csv')
print("Dataset Head:\n", df.head())
print("\nInfo:\n", df.info())
print("\nMissing Values:\n", df.isnull().sum())


def preprocess_data(df):
    """
    Preprocess the dataset by encoding categorical variables and normalizing numerical features.
    Args:
        df (pd.DataFrame): Raw dataset.
    Returns:
        pd.DataFrame: Preprocessed dataset.
    """
    # Encode 'Franchise' and 'Category'
    df['Franchise'] = LabelEncoder().fit_transform(df['Franchise'])
    df['Category'] = LabelEncoder().fit_transform(df['Category'])

    # Drop unnecessary columns
    df.drop(['ID', 'Name'], axis=1, inplace=True)
    return df

def normalize_features(df):
    """
    Normalize the numerical features in the dataset.
    Args:
        df (pd.DataFrame): Preprocessed dataset.
    Returns:
        pd.DataFrame: Dataset with normalized features.
    """
    scaler = StandardScaler()
    df[['No_of_item', 'Order_Placed']] = scaler.fit_transform(df[['No_of_item', 'Order_Placed']])
    return df

def train_model(X_train, y_train):
    """
    Train a linear regression model.
    Args:
        X_train (pd.DataFrame): Training feature set.
        y_train (pd.Series): Training target variable.
    Returns:
        model: Trained linear regression model.
    """
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    """
    Evaluate the model using RMSE and R-squared metrics.
    Args:
        model: Trained model.
        X_test (pd.DataFrame): Test feature set.
        y_test (pd.Series): True values for the test set.
    """
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    print(f"RMSE: {rmse}")
    print(f"R-Squared: {r2}")


def save_model(model, file_name=r'C:\\Python_Scripts\\Models\\restaurant_revenue_model.pkl'):
    """
    Save the trained model using joblib.
    Args:
        model: Trained model.
        file_name (str): Path where model will be saved.
    """
    joblib.dump(model, file_name)
    print(f"Model saved to {file_name}")

def load_model(file_name='C:\\Python_Scripts\\Models\\restaurant_revenue_model.pkl'):
    """
    Load a saved model.
    Args:
        file_name (str): Path to the model file.
    Returns:
        model: Loaded model.
    """
    return joblib.load(file_name)

def predict_new_data(model, new_data):
    """
    Predict the revenue for new restaurant data.
    Args:
        model: Trained model.
        new_data (pd.DataFrame): DataFrame with new observations.
    Returns:
        np.array: Predicted revenues.
    """
    predictions = model.predict(new_data)
    return predictions

if __name__ == "__main__":
    # Load the data
     # Preprocess the data
    df = preprocess_data(df)
    df = normalize_features(df)

    # Split the data into training and testing sets
    X = df.drop('Revenue', axis=1)  # Independent features
    y = df['Revenue']  # Dependent variable
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the model
    model = train_model(X_train, y_train)

    # Evaluate the model
    evaluate_model(model, X_test, y_test)

    # Save the trained model
    save_model(model)

    # Example: Predict revenue for a new restaurant observation
    model = load_model()
    new_data = pd.DataFrame({
        "Franchise": [1],
        "Category": [2],
        "No_of_item": [50],
        "Order_Placed": [30]
    })
    new_data = normalize_features(new_data)  # Normalize the new data
    print("Predicted Revenue:", predict_new_data(model, new_data))
