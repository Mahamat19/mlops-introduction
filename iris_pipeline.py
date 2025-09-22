# iris_pipeline.py
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
import numpy as np

def main():
    # Load dataset
    iris = load_iris(as_frame=True)
    X = iris.data
    y = iris.target

    print("First 5 rows of the dataset:")
    print(X.head())

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train model
    clf = RandomForestClassifier(random_state=42)
    clf.fit(X_train, y_train)

    # Evaluate
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print(f"\nModel accuracy: {acc:.2f}")

    # --- Logistic Regression model ---
    logreg_clf = LogisticRegression(max_iter=200, random_state=42)
    logreg_clf.fit(X_train, y_train)
    logreg_pred = logreg_clf.predict(X_test)
    logreg_acc = accuracy_score(y_test, logreg_pred)
    print(f"Logistic Regression accuracy: {logreg_acc:.2f}")

if __name__ == "__main__":
    main()
