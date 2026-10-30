import pandas as pd
import joblib
import json

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_val_score,
    GridSearchCV
)

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)


# 1. LOAD DATASET

df = pd.read_csv("student_placement.csv")

print("\nSTUDENT DATASET")
print(df)


# 2. SEPARATE FEATURES AND TARGET

X = df.drop("Placed", axis=1)
y = df["Placed"]


# 3. SPLIT DATASET

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# 4. MODEL COMPARISON

models = {
    "Logistic Regression": make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000)
    ),

    "Decision Tree": DecisionTreeClassifier(
        random_state=42
    ),

    "Random Forest": RandomForestClassifier(
        random_state=42
    )
}

print("\nMODEL COMPARISON")

for name, algorithm in models.items():

    algorithm.fit(X_train, y_train)

    predictions = algorithm.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    print(f"{name}: {accuracy:.2f}")


# 5. HYPERPARAMETER TUNING

param_grid = {
    "n_estimators": [50, 100, 150],
    "max_depth": [None, 3, 5],
    "min_samples_split": [2, 4]
}

grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid,
    cv=3,
    scoring="accuracy"
)

grid_search.fit(X_train, y_train)

model = grid_search.best_estimator_

print("\nBEST HYPERPARAMETERS")
print(grid_search.best_params_)

print("\nBEST GRID SEARCH SCORE")
print(grid_search.best_score_)


# 6. CROSS VALIDATION

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

cv_scores = cross_val_score(
    model,
    X,
    y,
    cv=cv,
    scoring="accuracy"
)

print("\nCROSS VALIDATION SCORES")
print(cv_scores)

print("\nAVERAGE CROSS VALIDATION ACCURACY")
print(cv_scores.mean())


# 7. FINAL MODEL TRAINING

model.fit(X_train, y_train)

y_pred = model.predict(X_test)


# 8. MODEL EVALUATION

accuracy = accuracy_score(y_test, y_pred)

precision = precision_score(
    y_test,
    y_pred,
    zero_division=0
)

recall = recall_score(
    y_test,
    y_pred,
    zero_division=0
)

f1 = f1_score(
    y_test,
    y_pred,
    zero_division=0
)

print("\nFINAL MODEL EVALUATION")

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)

print("\nCLASSIFICATION REPORT")
print(classification_report(y_test, y_pred, zero_division=0))


# 9. CONFUSION MATRIX

cm = confusion_matrix(y_test, y_pred)

print("\nCONFUSION MATRIX")
print(cm)

with open("confusion_matrix.json", "w") as file:
    json.dump(cm.tolist(), file)


# 10. SAVE METRICS

metrics = {
    "Accuracy": accuracy,
    "Precision": precision,
    "Recall": recall,
    "F1 Score": f1
}

with open("model_metrics.json", "w") as file:
    json.dump(metrics, file, indent=4)


# 11. SAVE FINAL MODEL

joblib.dump(model, "placement_model.pkl")

print("\nFINAL MODEL SAVED SUCCESSFULLY!")
print("Training process completed.")
