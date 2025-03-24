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

# Ensure model structure
if isinstance(model, Pipeline):
    preprocessor = model.named_steps.get('preprocessor', None)
    classifier = model.named_steps.get('classifier', model)
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

# Home Page
if app_mode == "🏠 Home":
    st.write("### Predict Credit Card Default")

    with st.form("user_input_form"):
        user_inputs = {col: st.number_input(col, value=0) for col in expected_columns}
        submitted = st.form_submit_button("Predict")

    if submitted:
        user_df = pd.DataFrame([user_inputs])
        X_transformed = preprocessor.transform(user_df) if preprocessor else user_df
        prediction = classifier.predict(X_transformed)
        probability = classifier.predict_proba(X_transformed)[:, 1]

        st.write("### Prediction Result")
        st.write(f"Default Risk: {'High' if prediction[0] == 1 else 'Low'}")
        st.write(f"Probability of Default: {probability[0]:.2f}")

    # CSV Upload for Batch Predictions
    st.write("#### Upload a CSV File for Predictions")
    uploaded_file = st.file_uploader("📂 Upload CSV", type=["csv"])

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        if set(expected_columns).issubset(df.columns):
            X_transformed = preprocessor.transform(df) if preprocessor else df
            predictions = classifier.predict(X_transformed)
            probabilities = classifier.predict_proba(X_transformed)[:, 1]

            df['Default_Risk'] = predictions
            df['Probability'] = probabilities

            st.write("### Prediction Results")
            st.dataframe(df[['LIMIT_BAL', 'AGE', 'SEX', 'EDUCATION', 'MARRIAGE', 'Default_Risk', 'Probability']])
        else:
            st.error("Uploaded CSV format is incorrect! Ensure correct columns.")

# Feature Importance Page
elif app_mode == "📊 Feature Importance":
    st.write("### 🔍 Feature Importance & Explainability")
    uploaded_file = st.file_uploader("📂 Upload CSV for SHAP Analysis", type=["csv"])

    if uploaded_file:
        df = pd.read_csv(uploaded_file)

        if set(expected_columns).issubset(df.columns):
            X_transformed = preprocessor.transform(df) if preprocessor else df
            sample_data = X_transformed[:5]

            explainer = shap.TreeExplainer(classifier)
            shap_values = explainer.shap_values(sample_data)

            correct_shap_values = shap_values[1] if isinstance(shap_values, list) else shap_values
            shap_importance = np.abs(correct_shap_values).mean(axis=0)
            
            importance_df = pd.DataFrame({'Feature': expected_columns, 'SHAP Importance': shap_importance})
            importance_df = importance_df.sort_values(by="SHAP Importance", ascending=False).head(10)

            st.write("### 🔥 Top 10 Most Important Features")
            st.dataframe(importance_df)

            fig, ax = plt.subplots(figsize=(8, 5))
            ax.barh(importance_df["Feature"], importance_df["SHAP Importance"], color="royalblue")
            ax.set_xlabel("SHAP Importance")
            ax.set_ylabel("Feature")
            ax.set_title("📊 Feature Importance")
            plt.gca().invert_yaxis()
            st.pyplot(fig)

            st.write("### 📊 SHAP Summary Plot")
            shap.summary_plot(correct_shap_values, sample_data, feature_names=expected_columns, show=False)
            plt.savefig("shap_summary.png", bbox_inches='tight')
            st.image("shap_summary.png")
        else:
            st.error("Uploaded CSV format is incorrect! Ensure correct columns.")
