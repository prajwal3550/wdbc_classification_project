# wdbc_classification.py

import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier


# --------------------------------------------------
# STEP 1: Load Dataset
# --------------------------------------------------

data = load_breast_cancer()

X = data.data
y = data.target

print("Data shape:", X.shape)
print("Target shape:", y.shape)
print("Target names:", data.target_names)
print("First 5 feature names:", data.feature_names[:5])

# --------------------------------------------------
# STEP 2: Train/Test Split
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

# --------------------------------------------------
# STEP 3 & 4: Train Multiple Classifiers
# --------------------------------------------------

classifiers = {
    "SVM": SVC(),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "KNN": KNeighborsClassifier(),
    "Random Forest": RandomForestClassifier(random_state=42),
}

results = {}

print("\nClassifier Results")
print("-" * 40)

for name, clf in classifiers.items():

    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)

    results[name] = accuracy

    print(f"{name} Accuracy : {accuracy:.4f}")
    print(f"{name} Precision: {precision:.4f}")
    print(f"{name} Recall   : {recall:.4f}")
    print()

# --------------------------------------------------
# STEP 5: Find Best Classifier
# --------------------------------------------------

best_classifier = max(results, key=results.get)

print("=" * 40)
print("Best Classifier:", best_classifier)
print("Best Accuracy:", round(results[best_classifier], 4))
print("=" * 40)

# Train best model again
best_model = classifiers[best_classifier]
best_model.fit(X_train, y_train)

y_pred_best = best_model.predict(X_test)

# --------------------------------------------------
# STEP 6: Confusion Matrix
# --------------------------------------------------

cm = confusion_matrix(y_test, y_pred_best)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=data.target_names,
)

disp.plot(cmap="Blues")
plt.title(f"{best_classifier} Confusion Matrix")
plt.savefig("wdbc_classification_matrix.png")
plt.close()

# --------------------------------------------------
# STEP 7: Scatter Plot
# --------------------------------------------------

plt.figure(figsize=(8, 6))

plt.scatter(
    X[:, 0],
    X[:, 1],
    c=y,
    cmap="coolwarm",
    edgecolors="k",
)

plt.xlabel(data.feature_names[0])
plt.ylabel(data.feature_names[1])
plt.title("WDBC Dataset Scatter Plot")

plt.savefig("wdbc_classification_scatter.png")
plt.close()

print("\nPlots saved successfully.")
print(" - wdbc_classification_matrix.png")
print(" - wdbc_classification_scatter.png")