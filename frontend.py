import streamlit as st
import pandas as pd
import requests
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.title("Customer Churn Prediction App")

# Allow user to specify API URL
api_url = st.text_input("API URL", value="http://localhost:8000/predict", help="Enter the Flask API endpoint, e.g., http://localhost:8000/predict")

# File upload
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file is not None:
    # Read the CSV for preview
    try:
        uploaded_file.seek(0)  # Reset cursor for preview
        df = pd.read_csv(uploaded_file)
        st.write("Uploaded Data Preview:")
        st.dataframe(df.head())
    except Exception as e:
        st.error(f"Error reading CSV: {str(e)}")
        logger.error(f"Error reading CSV: {str(e)}")
        st.stop()
    
    # Send file to API
    with st.spinner("Sending data to API..."):
        try:
            uploaded_file.seek(0)  # Reset cursor before sending
            files = {"file": (uploaded_file.name, uploaded_file, "text/csv")}
            logger.info(f"Sending request to {api_url}")
            response = requests.post(api_url, files=files, timeout=10)
            
            if response.status_code == 200:
                predictions = response.json()['predictions']
                pred_df = pd.DataFrame(predictions)
                st.write("Predictions:")
                st.dataframe(pred_df)
                
                csv = pred_df.to_csv(index=False)
                st.download_button(
                    label="Download Predictions",
                    data=csv,
                    file_name="churn_predictions.csv",
                    mime="text/csv"
                )
            else:
                st.error(f"API Error: {response.json().get('message', 'Unknown error')}")
                logger.error(f"API returned status {response.status_code}: {response.text}")
        except requests.exceptions.ConnectionError:
            st.error(f"Cannot connect to API at {api_url}. Is the Flask server running?")
            logger.error(f"Connection error: Cannot reach {api_url}")
        except requests.exceptions.Timeout:
            st.error(f"Request to {api_url} timed out. Check the API server.")
            logger.error(f"Timeout error: Request to {api_url}")
        except Exception as e:
            st.error(f"Error communicating with API: {str(e)}")
            logger.error(f"API request error: {str(e)}")