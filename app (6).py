import streamlit as st
import pandas as pd
import joblib
import gdown
import os
import numpy as np
import shap
import matplotlib.pyplot as plt

# Step 1: Download the model from Google Drive
model_url = "https://drive.google.com/uc?id=1en2IPj_z6OivZCBNDXepX-EAiZLvCILE"
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
        df = pd.read_csv(uploaded_file, header=None, names=expected_columns)

        if df.shape[1] != len(expected_columns):
            st.error("Uploaded CSV format is incorrect! Check the column count.")
        else:
            preprocessor = model.named_steps['preprocessor']
            classifier = model.named_steps['classifier']

            X_transformed = preprocessor.transform(df)
            predictions = classifier.predict(X_transformed)
            probabilities = classifier.predict_proba(X_transformed)[:, 1]

            df['Default_Risk'] = predictions
            df['Probability'] = probabilities

            st.write("### Prediction Results")
            st.dataframe(df[['LIMIT_BAL', 'AGE', 'SEX', 'EDUCATION', 'MARRIAGE', 'Default_Risk', 'Probability']])

elif app_mode == "📊 Feature Importance":
    st.write("### 🔍 Feature Importance & Explainability")

    uploaded_file = st.file_uploader("📂 Upload CSV for SHAP Analysis", type=["csv"])

    if uploaded_file:
        df = pd.read_csv(uploaded_file, header=None, names=expected_columns)

        if df.shape[1] != len(expected_columns):
            st.error("Uploaded CSV format is incorrect! Check the column count.")
        else:
            preprocessor = model.named_steps['preprocessor']
            classifier = model.named_steps['classifier']
            X_transformed = preprocessor.transform(df)

            feature_names = expected_columns  # Use original feature names
            sample_data = X_transformed[:5]  # Reduce sample size for speed

            explainer = shap.TreeExplainer(classifier)
            shap_values = explainer.shap_values(sample_data)

            # Ensure correct shape for SHAP values
            correct_shap_values = shap_values[1] if isinstance(shap_values, list) else shap_values
            shap_importance = np.abs(correct_shap_values).mean(axis=0)

            # Convert to 1D array
            shap_importance = np.array(shap_importance).flatten()

            # Ensure dimensions match
            min_len = min(len(feature_names), len(shap_importance))
            feature_names = feature_names[:min_len]
            shap_importance = shap_importance[:min_len]

            # Create DataFrame for feature importance
            importance_df = pd.DataFrame({'Feature': feature_names, 'SHAP Importance': shap_importance})
            importance_df = importance_df.sort_values(by="SHAP Importance", ascending=False).head(10)

            # Display results
            st.write("### 🔥 Top 10 Most Important Features")
            st.dataframe(importance_df)

            # Plot bar chart
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.barh(importance_df["Feature"], importance_df["SHAP Importance"], color="royalblue")
            ax.set_xlabel("SHAP Importance")
            ax.set_ylabel("Feature")
            ax.set_title("📊 Feature Importance")
            plt.gca().invert_yaxis()
            st.pyplot(fig)

            # SHAP Summary Plot
            st.write("### 📊 SHAP Summary Plot")
            shap.summary_plot(correct_shap_values, sample_data, feature_names=feature_names, show=False)
            plt.savefig("shap_summary.png", bbox_inches='tight')
            st.image("shap_summary.png")
