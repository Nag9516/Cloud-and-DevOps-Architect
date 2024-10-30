# heart_disease_prediction.py by Nagesha KS

# Import necessary libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score
import os
import graphviz
import os
os.environ["PATH"] += os.pathsep + r'C:\Program Files\Graphviz\bin'


# Define functions in the correct order
def load_dataset(path):
    return pd.read_csv(path)

def main():
   # dataset_path = r"C:\Python_Scripts\heart+disease\heart_disease.csv"  # Use raw string to handle backslashes
    dataset_path = "C:/Python_Scripts/heart+disease/heart_disease.csv"

    df = load_dataset(dataset_path)
    # Proceed with analysis, visualization, etc.


def analyze_dataset(df):
    # Example function to analyze data; define the full function as required
    print(df.describe())
    print(df.info())

def visualize_data(df):
    # Example visualization function; define your visualizations here
    sns.countplot(data=df, x="target")
    plt.show()


def main():
    # Path to dataset
   
       dataset_path = r"C:\Python_Scripts\heart+disease\heart_disease.csv"  # Raw string
    
    # Load data
    df = load_dataset(dataset_path)
      
  
# def load_dataset(r'C:\\Python_Scripts\\heart_disease.csv'):
    return pd.read_csv(r'C:\\Python_Scripts\\heart_disease.csv')

# Analyze dataset
def analyze_dataset(df):
    print("Dataset Overview:\n", df.describe())
    print("\nDataset Info:\n", df.info())
    print("\nCorrelation Matrix:\n", df.corr())
    return df.corr()
    

# Data visualization
def visualize_data(df):
    sns.countplot(data=df, x='target')
    plt.title("Patients with and without Heart Disease")
    plt.show()

    sns.scatterplot(data=df, x='age', y='target', hue='target')
    plt.title("Age vs Heart Disease Presence")
    plt.show()

    sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
    plt.title("Correlation Heatmap of Features")
    plt.show()

# Train-Test Split
def split_data(df):
    X = df.drop(columns=['target'])
    y = df['target']
    return train_test_split(X, y, test_size=0.3, random_state=42)

# Logistic Regression Model
def logistic_regression(X_train, X_test, y_train, y_test):
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    acc = accuracy_score(y_test, predictions)
    print("\nLogistic Regression Accuracy:", acc)
    print(confusion_matrix(y_test, predictions))
    return acc, predictions

# Decision Tree Model
def decision_tree(X_train, X_test, y_train, y_test):
    model = DecisionTreeClassifier(random_state=42)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    acc = accuracy_score(y_test, predictions)
    print("\nDecision Tree Accuracy:", acc)
    print(confusion_matrix(y_test, predictions))
    
    # Visualize decision tree
    dot_data = export_graphviz(model, out_file=None, feature_names=X_train.columns, class_names=["No Disease", "Disease"], filled=True)
    graph = graphviz.Source(dot_data)
    graph.render("decision_tree")
    return acc, predictions

# Random Forest Model
def random_forest(X_train, X_test, y_train, y_test):
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    acc = accuracy_score(y_test, predictions)
    print("\nRandom Forest Accuracy:", acc)
    print(confusion_matrix(y_test, predictions))
    
    # Visualize random forest
    dot_data = export_graphviz(model.estimators_[0], out_file=None, feature_names=X_train.columns, class_names=["No Disease", "Disease"], filled=True)
    graph = graphviz.Source(dot_data)
    graph.render("random_forest")
    return acc, predictions

# Evaluate all models and select the best
def evaluate_models(y_test, lr_preds, dt_preds, rf_preds):
    print("\nLogistic Regression Classification Report:\n", classification_report(y_test, lr_preds))
    print("\nDecision Tree Classification Report:\n", classification_report(y_test, dt_preds))
    print("\nRandom Forest Classification Report:\n", classification_report(y_test, rf_preds))

    # Print confusion matrices for heatmaps
    models = {"Logistic Regression": lr_preds, "Decision Tree": dt_preds, "Random Forest": rf_preds}
    for model_name, predictions in models.items():
        cm = confusion_matrix(y_test, predictions)
        sns.heatmap(cm, annot=True, fmt="d", cmap='Blues')
        plt.title(f"Confusion Matrix - {model_name}")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.show()

    # Calculate evaluation metrics
    metrics = {}
    for name, preds in models.items():
        precision = precision_score(y_test, preds)
        recall = recall_score(y_test, preds)
        f1 = f1_score(y_test, preds)
        acc = accuracy_score(y_test, preds)
        metrics[name] = {"Accuracy": acc, "Precision": precision, "Recall": recall, "F1 Score": f1}
    
    # Select best model based on accuracy
    best_model = max(metrics, key=lambda x: metrics[x]["Accuracy"])
    print("\nBest Model:", best_model)
    print("\nMetrics Summary:", metrics)
    return best_model, metrics

# Main execution function
def main():
    # Path to dataset
    dataset_path = "C:\Python_Scripts\heart+disease\heart_disease.csv"
    
    # Load data
    df = load_dataset(dataset_path)
    
    # Analyze data
    analyze_dataset(df)
    
    # Visualize data
    visualize_data(df)
    
    # Split data into training and test sets
    X_train, X_test, y_train, y_test = split_data(df)
    
    # Run models
    lr_acc, lr_preds = logistic_regression(X_train, X_test, y_train, y_test)
    dt_acc, dt_preds = decision_tree(X_train, X_test, y_train, y_test)
    rf_acc, rf_preds = random_forest(X_train, X_test, y_train, y_test)
    
    # Evaluate models
    evaluate_models(y_test, lr_preds, dt_preds, rf_preds)

# Run the main function
if __name__ == "__main__":
    main()
