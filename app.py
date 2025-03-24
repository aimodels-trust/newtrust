import streamlit as st
import pandas as pd
import joblib
import gdown
import os
import numpy as np
import shap
import matplotlib.pyplot as plt

# Step 1: Download the trained model from Google Drive
model_url = "https://drive.google.com/uc?id=1x4Vmmr6Ip-msXGQpeIa-WFkpyD5aECOo"
model_path = "credit_default_model.pkl"

if not os.path.exists(model_path):
    gdown.download(model_url, model_path, quiet=False)

# Step 2: Load the trained model
model = joblib.load(model_path)

# Step 3: Define Streamlit app
st.set_page_config(page_title="Credit Default Prediction", layout="wide")
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
    st.write("### Upload a CSV file for predictions")

    uploaded_file = st.file_uploader("📂 Upload CSV", type=["csv"])

    if uploaded_file:
        df = pd.read_csv(uploaded_file, header=None)

        # Check if CSV matches expected format
        if df.shape[1] != len(expected_columns):
            st.error(f"❌ CSV format incorrect! Expected {len(expected_columns)} columns, got {df.shape[1]}.")
        else:
            st.success("✅ File uploaded successfully!")

            # Assign column names to match the model's input
            df.columns = expected_columns

            # Reorder columns to match training data
            df = df[expected_columns]

            # Make predictions
            predictions = model.predict(df)
            probabilities = model.predict_proba(df)[:, 1]

            # Add predictions to DataFrame
            df['Default_Risk'] = predictions
            df['Probability'] = probabilities

            st.write("### Prediction Results")
            st.dataframe(df[['LIMIT_BAL', 'AGE', 'SEX', 'EDUCATION', 'MARRIAGE', 'Default_Risk', 'Probability']])

elif app_mode == "📊 Feature Importance":
    st.write("### 🔍 Feature Importance & Explainability")

    uploaded_file = st.file_uploader("📂 Upload CSV for SHAP Analysis", type=["csv"])

    if uploaded_file:
        df = pd.read_csv(uploaded_file, header=None)

        # Validate CSV format
        if df.shape[1] != len(expected_columns):
            st.error("❌ Incorrect CSV format! Please check the column count.")
        else:
            st.success("✅ CSV loaded successfully!")

            # Assign correct column names
            df.columns = expected_columns

            # Reorder columns
            df = df[expected_columns]

            # Ensure input transformation
            X_transformed = df.values

            # SHAP Explanation
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_transformed[:5])  # Use a small sample for speed

            # Extract SHAP importance values
            correct_shap_values = shap_values[1] if isinstance(shap_values, list) else shap_values
            shap_importance = np.abs(correct_shap_values).mean(axis=0)

            # Ensure dimensions match
            min_len = min(len(expected_columns), len(shap_importance))
            feature_names = expected_columns[:min_len]
            shap_importance = shap_importance[:min_len]

            # Create DataFrame for feature importance
            importance_df = pd.DataFrame({'Feature': feature_names, 'SHAP Importance': shap_importance})
            importance_df = importance_df.sort_values(by="SHAP Importance", ascending=False).head(10)

            # Display results
            st.write("### 🔥 Top 10 Most Important Features")
            st.dataframe(importance_df)

            # Plot feature importance
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.barh(importance_df["Feature"], importance_df["SHAP Importance"], color="royalblue")
            ax.set_xlabel("SHAP Importance")
            ax.set_ylabel("Feature")
            ax.set_title("📊 Feature Importance")
            plt.gca().invert_yaxis()
            st.pyplot(fig)

            # SHAP Summary Plot
            st.write("### 📊 SHAP Summary Plot")
            shap.summary_plot(correct_shap_values, X_transformed[:5], feature_names=feature_names, show=False)
            plt.savefig("shap_summary.png", bbox_inches='tight')
            st.image("shap_summary.png")
