
from pathlib import Path
import pickle

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier
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


def train_model():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)
    df = df.drop(columns=["Id"], errors="ignore")

    if not all(col in df.columns for col in FEATURES + ["Species"]):
        raise ValueError("Dataset is missing required columns.")

    X = df[FEATURES]
    y = df["Species"].map(LABELS)

    if y.isna().any():
        raise ValueError("Unknown species labels found.")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    scaler = MinMaxScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = DecisionTreeClassifier(
        max_depth=2,
        random_state=42
    )

    model.fit(X_train_scaled, y_train)

    train_predictions = model.predict(X_train_scaled)
    test_predictions = model.predict(X_test_scaled)

    train_accuracy = accuracy_score(
        y_train, train_predictions
    )

    test_accuracy = accuracy_score(
        y_test, test_predictions
    )

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)

    with open(SCALER_PATH, "wb") as f:
        pickle.dump(scaler, f)

    print("\nMODEL TRAINING SUCCESSFUL")
    print("-------------------------")
    print("Training samples:", len(X_train))
    print("Testing samples:", len(X_test))
    print(f"Training Accuracy: {train_accuracy * 100:.2f}%")
    print(f"Testing Accuracy: {test_accuracy * 100:.2f}%")

    print("\nClassification Report:")
    print(classification_report(
        y_test,
        model.predict(X_test_scaled),
        target_names=list(LABELS.keys()),
        zero_division=0
    ))

    print("Model saved:", MODEL_PATH)
    print("Scaler saved:", SCALER_PATH)

    return test_accuracy


if __name__ == "__main__":
    train_model()

