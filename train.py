
from pathlib import Path
import pickle
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

# Project directory
BASE_DIR = Path(__file__).resolve().parent

# Load dataset
df = pd.read_csv(BASE_DIR / "Iris - Iris.csv")
df = df.drop(columns=["Id"], errors="ignore")

# Separate features and target
X = df.drop(columns=["Species"])
y = df["Species"].map({
    "Iris-setosa": 0,
    "Iris-versicolor": 1,
    "Iris-virginica": 2
})

if y.isna().any():
    raise ValueError("Unknown species labels found in dataset.")

# Split: 80% training, 20% testing
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Scale features using training data only
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train model
model = DecisionTreeClassifier(
    max_depth=2,
    random_state=42
)
model.fit(X_train_scaled, y_train)

# Calculate accuracy
train_predictions = model.predict(X_train_scaled)
test_predictions = model.predict(X_test_scaled)

train_accuracy = accuracy_score(y_train, train_predictions)
test_accuracy = accuracy_score(y_test, test_predictions)

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))
print(f"Training Accuracy: {train_accuracy * 100:.2f}%")
print(f"Testing Accuracy: {test_accuracy * 100:.2f}%")

print("\nTraining Classification Report:")
print(classification_report(y_train, train_predictions, zero_division=0))

# Save model and scaler
with open(BASE_DIR / "Iris_Model.pkl", "wb") as f:
    pickle.dump(model, f)

with open(BASE_DIR / "Iris_Scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

print("\nModel saved: Iris_Model.pkl")
print("Scaler saved: Iris_Scaler.pkl")
