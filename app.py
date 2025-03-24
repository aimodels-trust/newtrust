import streamlit as st
import pandas as pd
import pickle
import os

st.title("Credit Default Prediction App")
st.write("Upload your trained model and a CSV file to get predictions.")

# Model uploader
model_file = st.file_uploader("Upload Model (.pkl)", type=["pkl"])

if model_file is not None:
    try:
        model = pickle.load(model_file)
        st.success("Model loaded successfully!")
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()
    
    # File uploader for CSV
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
