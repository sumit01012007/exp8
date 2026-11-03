import pandas as pd
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
df = pd.read_csv(BASE_DIR / "Iris - Iris.csv")

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)



# Basic inspection
print(df.head())
print(df.tail())
print("Shape:", df.shape)
print("Duplicates:", df.duplicated().sum())

# Remove unnecessary Id column
df = df.drop(columns=['Id'])

# Dataset information
df.info()

# Check class distribution
print(df['Species'].value_counts())

# Convert target labels into numbers
df['Species'] = df['Species'].map({
    'Iris-setosa': 0,
    'Iris-versicolor': 1,
    'Iris-virginica': 2
})

print(df['Species'])

# Separate features and target
x = df.drop('Species', axis=1)
y = df['Species']

print("X shape:", x.shape)
print("Y shape:", y.shape)

# Train-test split
x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.2,
    random_state=42
)

print("Training data:", x_train.shape)
print("Testing data:", x_test.shape)

# Feature scaling
scaler = MinMaxScaler()

# Fit ONLY on training data
scaler.fit(x_train)

print("Maximum:", scaler.data_max_)
print("Minimum:", scaler.data_min_)

# Transform both training and testing data
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

# Logistic Regression model
model = LogisticRegression()

# Train model
model.fit(x_train, y_train)

# Prediction
y_pred = model.predict(x_test)

print("Predicted values:")
print(y_pred)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Confusion matrix
print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(cm)
import pickle
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
iris = load_iris()
X = iris.data
y = iris.target
IrisPredModel = DecisionTreeClassifier()
IrisPredModel.fit(X, y)
with open("Iris_Model.pkl", "wb") as file:
    pickle.dump(IrisPredModel, file)
print("Model is saved as Iris_Model.pkl")