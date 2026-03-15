import os
import sys

# Add the project root directory to the Python path to resolve imports properly
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

import json
from src.preprocess import preprocess_text

def train_and_evaluate():
    # 1. Load Data
    data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'dataset.csv')
    print(f"Loading data from {data_path}...")
    try:
        df = pd.read_csv(data_path)
    except FileNotFoundError:
        print("Error: dataset.csv not found. Please ensure the path is correct.")
        return

    # 2. Preprocess Text
    print("Preprocessing text...")
    df['clean_comment'] = df['comment'].apply(preprocess_text)
    
    # Clean category labels (strip whitespace)
    df['category'] = df['category'].str.strip()

    # Filter out samples that became empty after preprocessing to avoid noise
    initial_len = len(df)
    df = df[df['clean_comment'].str.strip() != ""]
    print(f"Filtered {initial_len - len(df)} empty comments after preprocessing.")

    # 3. Train-Test Split
    X = df['clean_comment']
    y = df['category']
    
    # Ensure all classes are represented in split by using stratify
    # and use a larger test size if possible, though 0.25 is fine for ~200 rows.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    # 4. Feature Extraction (TF-IDF)
    print("Extracting features (TF-IDF)...")
    vectorizer = TfidfVectorizer()
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    # 5. Define Models
    models = {
        "Naive Bayes": MultinomialNB(),
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "SVM": SVC(kernel='linear')
    }

    best_model = None
    best_accuracy = 0.0
    best_model_name = ""

    # 6. Train and Evaluate Models
    print("\n--- Model Evaluation ---")
    for name, model in models.items():
        # Train
        model.fit(X_train_tfidf, y_train)
        
        # Predict
        y_pred = model.predict(X_test_tfidf)
        
        # Evaluate
        acc = accuracy_score(y_test, y_pred)
        print(f"\n{name} Accuracy: {acc:.4f}")
        print(f"Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}")
        try:
            print(f"Classification Report:\n{classification_report(y_test, y_pred, zero_division=0)}")
        except Exception:
            pass # Handle warning about zero division depending on sklearn version
        
        # Keep track of the best model
        if best_model is None or acc > best_accuracy:
            best_accuracy = acc
            best_model = model
            best_model_name = name

    print(f"\nBest Model: {best_model_name} with Accuracy: {best_accuracy:.4f}")

    # 7. Save the best model and vectorizer
    model_dir = os.path.join(os.path.dirname(__file__), '..', 'model')
    os.makedirs(model_dir, exist_ok=True)
    
    model_path = os.path.join(model_dir, 'classifier.pkl')
    vectorizer_path = os.path.join(model_dir, 'vectorizer.pkl')
    
    print(f"Saving best model to {model_path}...")
    joblib.dump(best_model, model_path)
    print(f"Saving vectorizer to {vectorizer_path}...")
    joblib.dump(vectorizer, vectorizer_path)
    
    # 8. Save metrics for the UI
    metrics_path = os.path.join(model_dir, 'metrics.json')
    class_dist = df['category'].value_counts().to_dict()
    metrics = {
        "best_model": best_model_name,
        "accuracy": best_accuracy,
        "dataset_size": len(df),
        "class_distribution": class_dist
    }
    print(f"Saving metrics to {metrics_path}...")
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=4)
    
    print("Training complete!")

if __name__ == "__main__":
    train_and_evaluate()
