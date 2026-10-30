import joblib
import pandas as pd

# Load the saved model

model = joblib.load("placement_model.pkl")

# Details of a new student

student = pd.DataFrame([{
"CGPA": 8.5,
"Aptitude_Score": 80,
"Communication_Score": 85,
"Internships": 2
}])

# Predict the placement class

prediction = model.predict(student)[0]

print("Student Details:")
print(student)

print("\nPredicted Placement Class:")

if prediction == 1:
    print("Placed")
else:
    print("Not Placed")
