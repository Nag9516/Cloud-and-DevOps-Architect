#Image Classification Using Scikit-learn.py by Nagesha KS

import os
import numpy as np
import pandas as pd
from skimage.io import imread
from skimage.transform import resize
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV

# Constants
INPUT_DIR = "C:\Python_Scripts\Cats_and_Dogs_Dataset"
CATEGORIES = ["Cat500", "Dog500"]
IMAGE_SIZE = (64, 64)  # Standard image size for classification

def load_images(input_dir, categories):
    """Load images from the specified directory and resize them."""
    data, labels = [], []
    for category_idx, category in enumerate(categories):
        category_path = os.path.join(input_dir, category)
        for file in os.listdir(category_path):
            img_path = os.path.join(category_path, file)
            img = imread(img_path)
            img_resized = resize(img, IMAGE_SIZE, anti_aliasing=True)
            data.append(img_resized.flatten())  # Flatten the image
            labels.append(category_idx)
    return np.asarray(data), np.asarray(labels)

def preprocess_data(data):
    """Standardize the data using StandardScaler."""
    scaler = StandardScaler()
    return scaler.fit_transform(data)

def evaluate_model(model, x_test, y_test):
    """Evaluate the model and print accuracy and classification report."""
    y_pred = model.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.4f}")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

def main():
    # Load data
    data, labels = load_images(INPUT_DIR, CATEGORIES)

    # Split the data into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

    # Preprocess the data
    x_train = preprocess_data(x_train)
    x_test = preprocess_data(x_test)

    # Model selection and training
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Decision Tree": DecisionTreeClassifier(),
        "Random Forest": RandomForestClassifier(n_estimators=100)
    }

    for model_name, model in models.items():
        print(f"Training {model_name}...")
        model.fit(x_train, y_train)
        evaluate_model(model, x_test, y_test)

    # Hyperparameter tuning for Random Forest
    rfc = RandomForestClassifier(random_state=42)
    param_grid = {
        'n_estimators': [200, 500],
        'max_features': ['auto', 'sqrt', 'log2'],
        'max_depth': [4, 5, 6, 7, 8],
        'criterion': ['gini', 'entropy']
    }
    
    #grid_search = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5)
    #print("Tuning Random Forest...")
    #grid_search.fit(x_train, y_train)
    #best_rfc = grid_search.best_estimator_

    #print("Best parameters found: ", grid_search.best_params_)
    #evaluate_model(best_rfc, x_test, y_test)

if __name__ == "__main__":
    main()
