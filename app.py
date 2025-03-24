import streamlit as st
import pandas as pd
import pickle
import os

# Load the trained model
MODEL_PATH = '/mnt/data/model (4).pkl'
if os.path.exists(MODEL_PATH):
    with open(MODEL_PATH, 'rb') as file:
        model = pickle.load(file)
else:
    st.error("Model file not found!")
    st.stop()

st.title("Credit Default Prediction App")
st.write("Upload a CSV file to get predictions.")

# File uploader
uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file is not None:
    try:
        # Read CSV
        data = pd.read_csv(uploaded_file)
        
        # Ensure input format (adjust as needed based on model expectations)
        st.write("### Uploaded Data Preview")
        st.write(data.head())
        
        # Make predictions
        predictions = model.predict(data)
        data['Prediction'] = predictions
        
        st.write("### Predictions")
        st.write(data)
        
        # Download button for results
        csv = data.to_csv(index=False).encode('utf-8')
        st.download_button("Download Predictions", data=csv, file_name="predictions.csv", mime="text/csv")
    
    except Exception as e:
        st.error(f"Error processing file: {e}")
