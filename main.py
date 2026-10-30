import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Step 1: Load the dataset
data = pd.read_csv("student_placement.csv")

# Step 2: Separate features and target
X = data.drop("Placed", axis=1)
y = data["Placed"]

# Step 3: Split the dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Step 4: Create both models
logistic_model = LogisticRegression(max_iter=1000)

random_forest_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

# Step 5: Train both models
logistic_model.fit(X_train, y_train)
random_forest_model.fit(X_train, y_train)

print("Both models trained successfully!")

# Step 6: Make predictions
logistic_predictions = logistic_model.predict(X_test)

random_forest_predictions = random_forest_model.predict(X_test)

# Step 7: Calculate accuracy
logistic_accuracy = accuracy_score(
    y_test, logistic_predictions
)

random_forest_accuracy = accuracy_score(
    y_test, random_forest_predictions
)

# Step 8: Compare the results
print("\n----- MODEL COMPARISON -----")

print(
    "Logistic Regression Accuracy:",
    round(logistic_accuracy * 100, 2), "%"
)

print(
    "Random Forest Accuracy:",
    round(random_forest_accuracy * 100, 2), "%"
)

# Step 9: Classification report for Random Forest
print("\n----- RANDOM FOREST REPORT -----")

print(classification_report(
    y_test,
    random_forest_predictions,
    labels=[0, 1],
    target_names=["Not Placed", "Placed"],
    zero_division=0
))