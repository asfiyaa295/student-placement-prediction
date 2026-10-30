import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# LOAD DATASET
df = pd.read_csv("student_placement.csv")
# DISPLAY DATASET
print("\nSTUDENT DATASET")
print(df)

# DATASET INFORMATION
print("\nDATASET INFORMATION")
df.info()

# STATISTICAL SUMMARY
print("\nSTATISTICAL SUMMARY")
print(df.describe())

# CHECK MISSING VALUES
print("\nMISSING VALUES")
print(df.isnull().sum())

# CHECK DUPLICATE RECORDS
print("\nDUPLICATE RECORDS")
print(df.duplicated().sum())

# PLACEMENT COUNT
print("\nPLACEMENT COUNT")
print(df["Placed"].value_counts())


# 1. PLACEMENT DISTRIBUTION CHART

placement_counts = df["Placed"].value_counts().sort_index()

plt.figure(figsize=(7, 5))
plt.bar(["Not Placed", "Placed"], placement_counts.values)

plt.title("Student Placement Distribution")
plt.xlabel("Placement Status")
plt.ylabel("Number of Students")

for i, value in enumerate(placement_counts.values):
    plt.text(i, value, str(value), ha="center")

plt.tight_layout()
plt.savefig("placement_distribution.png")
plt.close()


# 2. CGPA DISTRIBUTION HISTOGRAM

plt.figure(figsize=(8, 5))

plt.hist(df["CGPA"], bins=5, edgecolor="black")

plt.title("Distribution of Student CGPA")
plt.xlabel("CGPA")
plt.ylabel("Number of Students")

plt.tight_layout()
plt.savefig("cgpa_distribution.png")
plt.close()


# 3. CGPA VS PLACEMENT STATUS

plt.figure(figsize=(8, 5))

not_placed = df[df["Placed"] == 0]["CGPA"]
placed = df[df["Placed"] == 1]["CGPA"]

plt.boxplot(
    [not_placed, placed],
    tick_labels=["Not Placed", "Placed"]
)

plt.title("CGPA vs Placement Status")
plt.xlabel("Placement Status")
plt.ylabel("CGPA")

plt.tight_layout()
plt.savefig("cgpa_vs_placement.png")
plt.close()


# 4. INTERNSHIPS VS PLACEMENT STATUS

plt.figure(figsize=(8, 5))

not_placed = df[df["Placed"] == 0]["Internships"]
placed = df[df["Placed"] == 1]["Internships"]

plt.boxplot(
    [not_placed, placed],
    tick_labels=["Not Placed", "Placed"]
)

plt.title("Internships vs Placement Status")
plt.xlabel("Placement Status")
plt.ylabel("Number of Internships")

plt.tight_layout()
plt.savefig("internships_vs_placement.png")
plt.close()


# 5. CORRELATION HEATMAP

correlation = df.corr(numeric_only=True)

plt.figure(figsize=(9, 6))

sns.heatmap(
    correlation,
    annot=True,
    cmap="coolwarm",
    fmt=".2f",
    linewidths=0.5
)

plt.title("Student Placement Dataset - Correlation Heatmap")

plt.tight_layout()
plt.savefig("correlation_heatmap.png")
plt.close()


# 6. FEATURE IMPORTANCE ANALYSIS

model = joblib.load("placement_model.pkl")

importance = model.feature_importances_

feature_importance = pd.DataFrame({
    "Feature": [
        "CGPA",
        "Aptitude_Score",
        "Communication_Score",
        "Internships"
    ],
    "Importance": importance
})

feature_importance = feature_importance.sort_values(
    by="Importance",
    ascending=False
)

print("\nFEATURE IMPORTANCE")
print(feature_importance)

plt.figure(figsize=(8, 5))

sns.barplot(
    data=feature_importance,
    x="Importance",
    y="Feature"
)

plt.title("Feature Importance in Student Placement Prediction")
plt.tight_layout()

plt.savefig("feature_importance.png")
plt.close()

print("\nEDA COMPLETED SUCCESSFULLY!")
print("All charts saved in your project folder.")
