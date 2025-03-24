import streamlit as st
import pandas as pd
import pickle
import gdown

# Google Drive model file URL
MODEL_URL = "https://drive.google.com/uc?id=1x4Vmmr6Ip-msXGQpeIa-WFkpyD5aECOo"
MODEL_PATH = "model.pkl"

# Download and load the trained model
def load_model():
    try:
        gdown.download(MODEL_URL, MODEL_PATH, quiet=False)
        with open(MODEL_PATH, "rb") as f:
            model = pickle.load(f)
        return model
    except Exception as e:
        st.error("Error loading model: " + str(e))
        return None

# Load dataset
@st.cache_data
def load_data():
    file = st.file_uploader("Upload CSV dataset", type=["csv"])
    if file:
        df = pd.read_csv(file)
        return df
    return None

st.title("Credit Default Prediction")

df = load_data()
model = load_model()

if df is not None and model is not None:
    st.write("Dataset Preview:", df.head())
    
    # Ensure 'ID' column is dropped if present
    df = df.drop(columns=['ID'], errors='ignore')
    
    # Make predictions
    predictions = model.predict(df)
    df['Prediction'] = predictions
    
    st.write("Predictions:")
    st.dataframe(df)
