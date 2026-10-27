
from pathlib import Path
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

CSV_FILE = "iris.csv"
TARGET_COLUMN = "Species"

df = pd.read_csv(CSV_FILE)
df = df.drop(columns=["Id"], errors="ignore")

print("\nDataset Overview")
print(df.head())
print("\nDataset Shape:", df.shape)
print("\nColumns:", df.columns.tolist())
print("\nMissing Values:\n", df.isnull().sum())
print("\nClass Distribution:\n", df[TARGET_COLUMN].value_counts())

df = df.dropna().copy()

X = df.drop(columns=[TARGET_COLUMN])
y = df[TARGET_COLUMN]

X = X.apply(pd.to_numeric, errors="raise")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining Samples:", len(X_train))
print("Testing Samples:", len(X_test))

model = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression(max_iter=1000, random_state=42))
])

model.fit(X_train, y_train)

print("\nModel training completed successfully!")

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"\nModel Accuracy: {accuracy:.2%}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=0))

labels = sorted(y.unique())

matrix = confusion_matrix(y_test, y_pred, labels=labels)

print("\nConfusion Matrix:")
print(matrix)

results_dir = Path("results")
results_dir.mkdir(exist_ok=True)

report = classification_report(
    y_test,
    y_pred,
    labels=labels,
    output_dict=True,
    zero_division=0
)

with (results_dir / "metrics.json").open("w", encoding="utf-8") as file:
    json.dump({
        "accuracy": float(accuracy),
        "training_samples": len(X_train),
        "testing_samples": len(X_test),
        "features": X.columns.tolist(),
        "target_column": TARGET_COLUMN,
        "classification_report": report
    }, file, indent=4)

plt.figure(figsize=(8, 6))
sns.heatmap(
    matrix,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=labels,
    yticklabels=labels
)

plt.title("Iris Classification - Confusion Matrix")
plt.xlabel("Predicted Species")
plt.ylabel("Actual Species")
plt.tight_layout()
plt.savefig(results_dir / "confusion_matrix.png", dpi=150)
plt.show()

print("\nResults saved in the results folder.")
print("metrics.json")
print("confusion_matrix.png")

new_flower = pd.DataFrame(
    [[5.1, 3.5, 1.4, 0.2]],
    columns=X.columns
)

prediction = model.predict(new_flower)

print("\nNew Flower Measurements:")
print(new_flower)
print("Predicted Species:", prediction[0])
