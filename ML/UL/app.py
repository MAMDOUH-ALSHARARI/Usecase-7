import requests
import streamlit as st

# Set the title of the app
st.title("Welcome to fottball player clustering")



st.header("Welcome To The Checker")
minutes_played = st.number_input("Player Minutes Played : ", step=1)
current_value = st.number_input("Player Current Value : ", step=1)
award = st.number_input("Player award : ", step=1)

# Prediction button
if st.button("Predict Category"):
        # API request URL
        url = "https://usecase-7-0zj7.onrender.com/DBSCAN_Football_Players/predict"
        # Data for the POST request
        data = {

            "minutes_played": minutes_played,
            "current_value": current_value,
            "award": award,
            
        }

        # Send the POST request
        try:
            response = requests.post(url, json=data)
            response.raise_for_status()  # Check for request errors
            prediction = response.json()  # Parse JSON response

            if prediction['pred'] == 0:
                prediction = "High Value"
            elif prediction['pred'] == 1:
                prediction = "Normal Value"
                
            st.write(f"The Value Of The Player : {prediction}")
        except requests.exceptions.RequestException as e:
            st.error("Error requesting prediction from API. Please try again.")
            st.write(e)