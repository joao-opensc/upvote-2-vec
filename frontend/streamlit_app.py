import streamlit as st
# from annotated_text import annotated_text
from datetime import datetime, time, timedelta
import os

import torch
import numpy as np

# from utils import *
# from model.architectures import *
# from model.model_loader import load_model

import requests

def main():

    st.title("Hacker News Upvote Predictor 🚀")

    input_container = st.container()
    with input_container:

        st.write("Enter post details below to predict the number of upvotes.")

        default_date = datetime.now().date()
        default_time = datetime.now()+timedelta(hours=1)
        # st.write(default_time.strftime("%H:%M"))

        title = st.text_input("Post Title")
        author = st.text_input("Author")
        url = st.text_input("URL Link Attached")
        date = st.date_input("Post Date", value=default_date)
        time_str = st.text_input("Post Time (HH:MM)", value=default_time.strftime("%H:%M"))



        st.write(f"You selected: {date} @ {time_str}")

        if st.button("Predict"):
            try:
                entered_time = datetime.strptime(time_str, "%H:%M").time()
                st.success(f"You selected: {date} @ {time_str}")
                # Combine into a single datetime object to convert to unix timestamp
                combined_datetime = f"{date} {time_str}:00"
                
                # Convert to datetime object
                dt = datetime.strptime(combined_datetime, "%Y-%m-%d %H:%M:%S")
                unix_timestamp = int(dt.timestamp())

                input_data = {
                    "title": title,
                    "url": url,
                    "user": author,
                    # "date": date,
                    "timestamp": unix_timestamp
                }
            except ValueError:
                st.error("Please enter time in 24-hour HH:MM format.")
            # prediction = model.predict(process_input(input_data))
            # st.metric("Predicted Upvotes", round(prediction[0], 2))
            
            try:
                backend_url = os.getenv("BACKEND_URL", "http://localhost:8888")
                response = requests.post(f"{backend_url}/predict", json=input_data)

                if response.status_code == 200:
                    prediction = response.json()["predicted_score"]
                    st.success(f"Predicted Upvotes: {prediction}")
                else:
                    st.error(f"Error from server: {response.text}")
            except Exception as e:
                st.error(f"Failed to connect to FastAPI: {e}")
main()