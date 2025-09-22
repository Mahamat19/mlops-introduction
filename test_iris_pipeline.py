# test_iris_pipeline.py
import pytest
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

# --- Dummy Tests to Demonstrate Testing in ML Pipelines ---

def test_load_data():
    iris = load_iris(as_frame=True)
    assert iris.data.shape[0] > 0  # dataset is not empty
    assert iris.target.nunique() == 3  # 3 classes

def test_train_test_split():
    iris = load_iris(as_frame=True)
    X_train, X_test, y_train, y_test = train_test_split(
        iris.data, iris.target, test_size=0.2, random_state=42
    )
    assert len(X_train) > len(X_test)  # training set larger than test set
    assert len(y_train) == len(X_train)

def test_random_forest_training():
    iris = load_iris(as_frame=True)
    X_train, X_test, y_train, y_test = train_test_split(
        iris.data, iris.target, test_size=0.2, random_state=42
    )
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    acc = model.score(X_test, y_test)
    assert acc > 0.5  # dummy check: accuracy should be better than random

def test_logistic_regression_training():
    iris = load_iris(as_frame=True)
    X_train, X_test, y_train, y_test = train_test_split(
        iris.data, iris.target, test_size=0.2, random_state=42
    )
    model = LogisticRegression(max_iter=200, random_state=42)
    model.fit(X_train, y_train)
    acc = model.score(X_test, y_test)
    assert acc > 0.5  # dummy check
