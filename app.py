import streamlit as st
import pandas as pd
import joblib
import gdown
import os
import numpy as np
import shap
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline

# Step 0: Set page configuration
st.set_page_config(page_title="Credit Default Prediction", layout="wide")

# Step 1: Download the model from Google Drive
model_url = "https://drive.google.com/uc?id=1x4Vmmr6Ip-msXGQpeIa-WFkpyD5aECOo"
model_path = "credit_default_model.pkl"

if not os.path.exists(model_path):
    gdown.download(model_url, model_path, quiet=False)

# Step 2: Load the trained model
@st.cache_resource
def load_model():
    return joblib.load(model_path)

model = load_model()

# Ensure model is a pipeline
if isinstance(model, Pipeline):
    preprocessor = model.named_steps['preprocessor']
    classifier = model.named_steps['classifier']
else:
    preprocessor = None
    classifier = model

# Step 3: Define Streamlit app
st.title("💳 Credit Card Default Prediction with Explainability")

# Sidebar Navigation
st.sidebar.title("Navigation")
app_mode = st.sidebar.radio("Select Mode", ["🏠 Home", "📊 Feature Importance"])

# Expected input features
expected_columns = [
    'LIMIT_BAL', 'SEX', 'EDUCATION', 'MARRIAGE', 'AGE',
    'PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6',
    'BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6',
    'PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6'
]

if app_mode == "🏠 Home":
    st.write("### Predict Credit Card Default")

    with st.form("user_input_form"):
        limit_bal = st.number_input("Credit Limit (LIMIT_BAL)", min_value=0)
        age = st.number_input("Age (AGE)", min_value=18, max_value=100)
        sex = st.selectbox("Sex (SEX)", options=[1, 2], format_func=lambda x: "Male" if x == 1 else "Female")
        education = st.selectbox("Education (EDUCATION)", options=[1, 2, 3, 4], format_func=lambda x: {1: "Graduate", 2: "University", 3: "High School", 4: "Others"}[x])
        marriage = st.selectbox("Marriage (MARRIAGE)", options=[1, 2, 3], format_func=lambda x: {1: "Married", 2: "Single", 3: "Others"}[x])
        pay_status = [st.number_input(f"Repayment Status (PAY_{i})", min_value=-2, max_value=8) for i in [0, 2, 3, 4, 5, 6]]
        bill_amt = [st.number_input(f"Bill Amount {i+1} (BILL_AMT{i+1})", min_value=0) for i in range(6)]
        pay_amt = [st.number_input(f"Payment Amount {i+1} (PAY_AMT{i+1})", min_value=0) for i in range(6)]
        submitted = st.form_submit_button("Predict")
    
    if submitted:
        user_data = pd.DataFrame([[limit_bal, sex, education, marriage, age] + pay_status + bill_amt + pay_amt], columns=expected_columns)
        X_transformed = preprocessor.transform(user_data) if preprocessor else user_data
        prediction = classifier.predict(X_transformed)
        probability = classifier.predict_proba(X_transformed)[:, 1]

        st.write("### Prediction Result")
        st.write(f"Default Risk: {'High' if prediction[0] == 1 else 'Low'}")
        st.write(f"Probability of Default: {probability[0]:.2f}")

elif app_mode == "📊 Feature Importance":
    st.write("### 🔍 Feature Importance & Explainability")
    uploaded_file = st.file_uploader("📂 Upload CSV for SHAP Analysis", type=["csv"])
    
    if uploaded_file:
        df = pd.read_csv(uploaded_file, header=None, names=expected_columns)
        if df.shape[1] != len(expected_columns):
            st.error("Uploaded CSV format is incorrect! Check the column count.")
        else:
            X_transformed = preprocessor.transform(df) if preprocessor else df
            explainer = shap.TreeExplainer(classifier)
            shap_values = explainer.shap_values(X_transformed[:5])
            shap_importance = np.abs(shap_values[1]).mean(axis=0) if isinstance(shap_values, list) else np.abs(shap_values).mean(axis=0)
            importance_df = pd.DataFrame({'Feature': expected_columns, 'SHAP Importance': shap_importance}).sort_values(by="SHAP Importance", ascending=False).head(10)
            
            st.write("### 🔥 Top 10 Most Important Features")
            st.dataframe(importance_df)
            
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.barh(importance_df["Feature"], importance_df["SHAP Importance"], color="royalblue")
            ax.set_xlabel("SHAP Importance")
            ax.set_ylabel("Feature")
            ax.set_title("📊 Feature Importance")
            plt.gca().invert_yaxis()
            st.pyplot(fig)
