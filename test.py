
from pathlib import Path
import pickle
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

BASE_DIR = Path(__file__).resolve().parent

# Load dataset
df = pd.read_csv(BASE_DIR / "Iris - Iris.csv")
df = df.drop(columns=["Id"], errors="ignore")

# Prepare features and target
X = df.drop(columns=["Species"])
y = df["Species"].map({
    "Iris-setosa": 0,
    "Iris-versicolor": 1,
    "Iris-virginica": 2
})

if y.isna().any():
    raise ValueError("Unknown species labels found in dataset.")

# Reproduce the same split used in train.py
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Load trained model and scaler
with open(BASE_DIR / "Iris_Model.pkl", "rb") as f:
    model = pickle.load(f)

with open(BASE_DIR / "Iris_Scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# Transform and predict
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

train_predictions = model.predict(X_train_scaled)
test_predictions = model.predict(X_test_scaled)

# Calculate accuracy
train_accuracy = accuracy_score(y_train, train_predictions)
test_accuracy = accuracy_score(y_test, test_predictions)

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))
print(f"Training Accuracy: {train_accuracy * 100:.2f}%")
print(f"Testing Accuracy: {test_accuracy * 100:.2f}%")

print("\nTest Classification Report:")
print(classification_report(
    y_test,
    test_predictions,
    target_names=[
        "Iris-setosa",
        "Iris-versicolor",
        "Iris-virginica"
    ],
    zero_division=0
))

# Assertion: test accuracy should not exceed training accuracy
# This checks your desired condition without crashing the script.
if test_accuracy < train_accuracy:
    print("\nAssertion passed: Test accuracy is lower than training accuracy.")
else:
    print(
        "\nAssertion condition not met: Test accuracy is equal to "
        "or greater than training accuracy."
    )

