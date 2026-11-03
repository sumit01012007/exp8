
from pathlib import Path
import pickle

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

BASE_DIR = Path(__file__).resolve().parent

DATA_PATH = BASE_DIR / "Iris - Iris.csv"
MODEL_PATH = BASE_DIR / "Iris_Model.pkl"
SCALER_PATH = BASE_DIR / "Iris_Scaler.pkl"

FEATURES = [
    "SepalLengthCm",
    "SepalWidthCm",
    "PetalLengthCm",
    "PetalWidthCm"
]

LABELS = {
    "Iris-setosa": 0,
    "Iris-versicolor": 1,
    "Iris-virginica": 2
}


def test_model():

    assert DATA_PATH.exists(), "Dataset missing"
    assert MODEL_PATH.exists(), "Model file missing"
    assert SCALER_PATH.exists(), "Scaler file missing"

    df = pd.read_csv(DATA_PATH)
    df = df.drop(columns=["Id"], errors="ignore")

    X = df[FEATURES]
    y = df["Species"].map(LABELS)

    assert not y.isna().any(), "Invalid species labels"

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)

    with open(SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)

    X_test_scaled = scaler.transform(X_test)

    predictions = model.predict(X_test_scaled)

    accuracy = accuracy_score(y_test, predictions)

    print("\nMODEL TESTING RESULTS")
    print("---------------------")
    print("Total test samples:", len(X_test))
    print("Correct predictions:", (predictions == y_test).sum())
    print(f"Accuracy: {accuracy * 100:.2f}%")

    print("\nClassification Report:")
    print(classification_report(
        y_test,
        predictions,
        target_names=list(LABELS.keys()),
        zero_division=0
    ))

    assert len(predictions) == len(y_test), "Prediction count mismatch"
    assert accuracy >= 0.80, "Model accuracy is below 80%"

    print("\nALL TESTS PASSED!")


if __name__ == "__main__":
    test_model()


