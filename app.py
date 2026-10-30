
import streamlit as st
import io
import pandas as pd
import joblib
import streamlit as st
import pandas as pd
import joblib
import json
import matplotlib.pyplot as plt
# ---------------- PAGE CONFIGURATION ----------------

st.set_page_config(
    page_title="Student Placement Prediction",
    page_icon="🎓",
    layout="wide"
)

# ---------------- LOAD MODEL ----------------

model = joblib.load("placement_model.pkl")
with open("model_metrics.json", "r") as file:
    metrics = json.load(file)
with open("confusion_matrix.json", "r") as file:
    cm = json.load(file)   

# ---------------- CUSTOM CSS ----------------

st.markdown("""
<style>

.stApp {
    background-color: #f4f7fc;
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #172554, #312e81);
}

[data-testid="stSidebar"] * {
    color: white;
}

.main-title {
    font-size: 36px;
    font-weight: 800;
    color: #172554;
}

.subtitle {
    color: #64748b;
    font-size: 16px;
}

.info-box {
    background: #e8f2ff;
    padding: 15px;
    border-radius: 12px;
    color: #1e40af;
    margin: 15px 0;
    border: 1px solid #bfdbfe;
}

.result-box {
    background: #dcfce7;
    padding: 25px;
    border-radius: 15px;
    text-align: center;
    border: 1px solid #86efac;
}

.result-title {
    color: #166534;
    font-size: 25px;
    font-weight: bold;
}

.stButton > button {
    background: linear-gradient(90deg, #4f46e5, #6366f1);
    color: white;
    border-radius: 10px;
    border: none;
    padding: 12px;
    font-weight: bold;
    width: 100%;
}

.stButton > button:hover {
    background: #3730a3;
    color: white;
}

.footer {
    text-align: center;
    color: #64748b;
    padding: 25px;
    font-size: 14px;
}

</style>
""", unsafe_allow_html=True)

# ---------------- SIDEBAR ----------------

with st.sidebar:
    st.markdown("# 🎓 Placement AI")
    st.markdown("---")

    st.markdown("### Navigation")
    st.markdown("🏠 Home")
    st.markdown("📘 About Project")
    st.markdown("🤖 Machine Learning Model")
    st.markdown("📊 Prediction")

    st.markdown("---")

    st.markdown("### Project Information")
    st.write("Technology: Python")
    st.write("ML: Random Forest")
    st.write("Interface: Streamlit")

    st.markdown("---")
    st.markdown("*Better Skills, Brighter Future*")

# ---------------- HEADER ----------------

st.markdown(
    '<div class="main-title">🎓 Student Placement Prediction</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Predict student placement class using Machine Learning</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="info-box">ℹ️ Educational demonstration only. '
    'Predictions are not guarantees of employment.</div>',
    unsafe_allow_html=True
)

st.write("")

# ---------------- MAIN LAYOUT ----------------

left, right = st.columns([1, 1])

# ---------------- STUDENT INPUT ----------------

with left:

    with st.container(border=True):

        st.subheader("👨‍🎓 Student Details")
        st.write("Enter the student's academic information.")

        cgpa = st.number_input(
            "CGPA",
            min_value=0.0,
            max_value=10.0,
            value=7.0,
            step=0.1
        )

        aptitude = st.number_input(
            "Aptitude Score",
            min_value=0,
            max_value=100,
            value=70
        )

        communication = st.number_input(
            "Communication Score",
            min_value=0,
            max_value=100,
            value=70
        )

        internships = st.number_input(
            "Number of Internships",
            min_value=0,
            max_value=10,
            value=1
        )

        predict = st.button(
            "✨ Predict Placement",
            use_container_width=True
        )

# ---------------- PREDICTION RESULT ----------------

with right:

    with st.container(border=True):

        st.subheader("📊 Prediction Result")

        if predict:

            student = pd.DataFrame([{
                "CGPA": cgpa,
                "Aptitude_Score": aptitude,
                "Communication_Score": communication,
                "Internships": internships
            }])

            prediction = model.predict(student)[0]
            probabilities = model.predict_proba(student)[0]

            st.session_state["prediction"] = int(prediction)
            st.session_state["student"] = student
            st.session_state["probabilities"] = probabilities.tolist()

        if "prediction" in st.session_state:

            prediction = st.session_state["prediction"]
            student = st.session_state["student"]
            probabilities = st.session_state["probabilities"]

            placed_probability = probabilities[1] * 100
            not_placed_probability = probabilities[0] * 100

            if prediction == 1:

                st.markdown(
                    '<div class="result-box">'
                    '<div style="font-size:45px">✅</div>'
                    '<div class="result-title">Predicted Class: Placed</div>'
                    '</div>',
                    unsafe_allow_html=True
                )

            else:

                st.warning("Predicted Class: Not Placed")

            st.markdown("### 📈 Prediction Probability")

            st.write(f"Placed: {placed_probability:.2f}%")
            st.progress(float(probabilities[1]))

            st.write(f"Not Placed: {not_placed_probability:.2f}%")
            st.progress(float(probabilities[0]))

            st.write("")
            st.markdown("### 👤 Student Summary")

            st.dataframe(
                student.rename(columns={
                    "CGPA": "CGPA",
                    "Aptitude_Score": "Aptitude",
                    "Communication_Score": "Communication",
                    "Internships": "Internships"
                }).T.rename(columns={0: "Value"}),
                use_container_width=True
            )

            st.info(
                "This is an experimental prediction from a "
                "small, invented dataset. It must not be used "
                "for actual employment decisions."
            )

        else:

            st.info(
                "Enter student details and click Predict Placement "
                "to see the result here."
            )

            st.markdown("### 🚀 Your Future Starts Here")

            st.write(
                "Machine Learning can identify patterns "
                "in historical student data."
            )
            # ---------------- FOOTER ----------------

st.markdown("---")
# ABOUT PROJECT SECTION

st.markdown("---")
st.header("📘 About This Project")
# CONFUSION MATRIX VISUALIZATION

st.markdown("---")
st.header("🔍 Confusion Matrix")

fig, ax = plt.subplots()

ax.imshow(cm, cmap="Blues")

ax.set_title("Student Placement Prediction")
ax.set_xlabel("Predicted Label")
ax.set_ylabel("Actual Label")

ax.set_xticks([0, 1])
ax.set_yticks([0, 1])

ax.set_xticklabels(["Not Placed", "Placed"])
ax.set_yticklabels(["Not Placed", "Placed"])

for i in range(2):
    for j in range(2):
        ax.text(j, i, cm[i][j], ha="center", va="center", fontsize=14)

st.pyplot(fig)
plt.close(fig)

st.caption(
    "This confusion matrix is based on only 3 test records "
    "from a small invented dataset."
)
# MODEL PERFORMANCE SECTION

st.markdown("---")
st.header("📊 Machine Learning Model Performance")

st.write("Evaluation results from the model's test dataset.")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Accuracy", f"{metrics['Accuracy'] * 100:.2f}%")

with col2:
    st.metric("Precision", f"{metrics['Precision'] * 100:.2f}%")

with col3:
    st.metric("Recall", f"{metrics['Recall'] * 100:.2f}%")

with col4:
    st.metric("F1 Score", f"{metrics['F1 Score'] * 100:.2f}%")

st.caption(
    "Note: These scores are based on only 3 test records "
    "from a small invented dataset. They do not demonstrate "
    "real-world prediction performance."
)

st.write(
    "Student Placement Prediction is a Machine Learning project "
    "that predicts a student's placement class based on academic "
    "and skill-related information."
)

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("### 🐍 Python")
    st.write("Used as the main programming language.")

with col2:
    st.markdown("### 🌳 Random Forest")
    st.write("Used to train the machine learning prediction model.")

with col3:
    st.markdown("### 🎨 Streamlit")
    st.write("Used to create the interactive web application.")

st.subheader("⚙️ How Does It Work?")

st.write("""
1. Collect student information.
2. Prepare the dataset.
3. Train a Random Forest model.
4. Save the trained model using Joblib.
5. Load the model into Streamlit.
6. Enter student details and generate a prediction.
""")

st.subheader("🧠 Machine Learning Algorithm")

st.write(
    "Random Forest combines predictions from multiple decision "
    "trees to classify students."
)

st.subheader("⚠️ Project Limitation")

st.write(
    "This is an educational prototype trained on a small, invented "
    "dataset. Its predictions are not reliable employment forecasts."
)


# YOUR EXISTING FOOTER — KEEP THIS

st.markdown("---")
# ABOUT PROJECT SECTION

st.markdown("---")
st.header("📘 About This Project")

st.write(
    "Student Placement Prediction is a Machine Learning project "
    "that predicts a student's placement class based on academic "
    "and skill-related information."
)

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("### 🐍 Python")
    st.write("Used as the main programming language.")

with col2:
    st.markdown("### 🌳 Random Forest")
    st.write("Used to train the machine learning prediction model.")

with col3:
    st.markdown("### 🎨 Streamlit")
    st.write("Used to create the interactive web application.")

st.subheader("⚙️ How Does It Work?")

st.write("""
1. Collect student information.
2. Prepare the dataset.
3. Train a Random Forest model.
4. Save the trained model using Joblib.
5. Load the model into Streamlit.
6. Enter student details and generate a prediction.
""")

st.subheader("🧠 Machine Learning Algorithm")

st.write(
    "Random Forest combines predictions from multiple decision "
    "trees to classify students."
)

st.subheader("⚠️ Project Limitation")

st.write(
    "This is an educational prototype trained on a small, invented "
    "dataset. Its predictions are not reliable employment forecasts."
)


# YOUR EXISTING FOOTER — KEEP THIS

st.markdown("---")

st.markdown(
    '<div class="footer">'
    '🎓 Student Placement Prediction | Built with Python, '
    'Scikit-learn and Streamlit'
    '</div>',
    unsafe_allow_html=True
)
# BATCH STUDENT PREDICTION

st.divider()

st.header("📂 Batch Student Placement Prediction")

st.write("Upload a CSV file to predict placement for multiple students.")

uploaded_file = st.file_uploader(
    "Upload Student CSV",
    type=["csv"],
    key="batch_upload"
)

if uploaded_file is not None:

    try:
        batch_df = pd.read_csv(uploaded_file)

        required_columns = [
            "CGPA",
            "Aptitude_Score",
            "Communication_Score",
            "Internships"
        ]

        # CHECK EMPTY DATASET
        if batch_df.empty:
            st.error("The uploaded CSV file is empty.")

        # CHECK REQUIRED COLUMNS
        elif not all(column in batch_df.columns for column in required_columns):
            st.error("CSV must contain all four required columns.")

        else:
            student_data = batch_df[required_columns].copy()

            # CHECK MISSING VALUES
            if student_data.isnull().any().any():
                st.error("Please remove missing values from your CSV file.")

            # CHECK NUMERIC VALUES
            elif not all(
                pd.api.types.is_numeric_dtype(student_data[col])
                for col in required_columns
            ):
                st.error("All student input values must be numeric.")

            # CHECK CGPA RANGE
            elif not student_data["CGPA"].between(0, 10).all():
                st.error("CGPA must be between 0 and 10.")

            # CHECK SCORE RANGE
            elif not student_data["Aptitude_Score"].between(0, 100).all():
                st.error("Aptitude score must be between 0 and 100.")

            elif not student_data["Communication_Score"].between(0, 100).all():
                st.error("Communication score must be between 0 and 100.")

            # CHECK INTERNSHIPS
            elif (
                (student_data["Internships"] < 0).any()
                or (student_data["Internships"] % 1 != 0).any()
            ):
                st.error("Internships must be non-negative whole numbers.")

            else:
                # MAKE PREDICTIONS
                batch_predictions = model.predict(student_data)

                probabilities = model.predict_proba(student_data)

                batch_df["Prediction"] = [
                    "Placed" if prediction == 1 else "Not Placed"
                    for prediction in batch_predictions
                ]

                batch_df["Placed_Probability"] = (
                    probabilities[:, 1] * 100
                ).round(2)

                st.success("Batch predictions completed!")

                st.dataframe(
                    batch_df,
                    use_container_width=True
                )

                # DOWNLOAD RESULTS
                csv_data = batch_df.to_csv(index=False).encode("utf-8")

                st.download_button(
                    label="⬇️ Download Predictions",
                    data=csv_data,
                    file_name="student_placement_predictions.csv",
                    mime="text/csv"
                )

                st.info(
                    "These are experimental predictions from a small, "
                    "invented dataset. They should not be used for actual "
                    "employment decisions."
                )

    except (pd.errors.ParserError, UnicodeDecodeError):
        st.error("Unable to read this CSV file. Please upload a valid CSV.")
