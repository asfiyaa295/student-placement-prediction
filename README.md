# Student Placement Prediction System

## Project Overview

The Student Placement Prediction System is a Machine Learning application that predicts student placement outcomes based on academic performance and skill-related information.

The application supports individual and batch predictions through an interactive Streamlit dashboard.

## Features

* Individual student placement prediction
* Batch prediction using CSV uploads
* Placement probability estimation
* Input validation
* Exploratory Data Analysis (EDA)
* Data visualization
* Correlation heatmap
* Feature importance analysis
* Machine Learning model evaluation
* Downloadable prediction results

## Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn
* Streamlit
* Joblib

## Machine Learning Techniques

* Logistic Regression
* Decision Tree Classifier
* Random Forest Classifier
* Train-Test Split
* Stratified K-Fold Cross-Validation
* GridSearchCV
* Hyperparameter Tuning
* Confusion Matrix
* Accuracy, Precision, Recall and F1 Score

## Dataset Features

| Feature             | Description                    |
| ------------------- | ------------------------------ |
| CGPA                | Student academic performance   |
| Aptitude_Score      | Aptitude test score            |
| Communication_Score | Communication assessment score |
| Internships         | Number of internships          |
| Placed              | Placement outcome (0 or 1)     |

## Project Structure

```text
Student_Placement_Prediction/
│
├── student_placement.csv
├── batch_students.csv
├── main.py
├── train_model.py
├── predict.py
├── eda.py
├── app.py
├── placement_model.pkl
├── model_metrics.json
├── confusion_matrix.json
├── requirements.txt
└── README.md
```

## Installation and Setup

1. Clone the repository.

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Train the model:

```bash
python train_model.py
```

4. Start the application:

```bash
python -m streamlit run app.py
```

## Disclaimer

This project currently uses a small, manually created dataset containing 15 invented student records. Predictions and evaluation metrics are experimental and are not suitable for actual employment decisions.

## Future Improvements

* Use a larger, representative real-world dataset.
* Improve model evaluation and generalization.
* Add more student-related features.
* Deploy the application online.
* Improve dashboard functionality.

## Author

B.Tech Artificial Intelligence and Machine Learning Student
