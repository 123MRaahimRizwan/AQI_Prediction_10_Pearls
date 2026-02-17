# import streamlit as st
# import pandas as pd
# import numpy as np

# import joblib
# from pymongo import MongoClient
# from pymongo.server_api import ServerApi

# st.set_page_config(page_title="AQI Prediction", layout="centered")

# st.title("🌫 AQI Prediction Dashboard - Karachi")

# # -----------------------------
# # Load model
# # -----------------------------
# model = joblib.load("aqi_model.pkl")

# # -----------------------------
# # MongoDB connection
# # -----------------------------
# MONGO_URI = "mongodb+srv://rizwan4601146_db_user:5G1bLMeOqUeWviQP@10pearlsaqicluster.rrn5bw6.mongodb.net/?appName=10PearlsAQICluster"
# # MONGO_URI = "mongodb+srv://rizwan4601146_db_user:5G1bLMeOqUeWviQP@10pearlsaqicluster.rrn5bw6.mongodb.net/?appName=10PearlsAQICluster"
# # mongodb+srv://rizwan4601146_db_user:<db_password>@10pearlsaqicluster.rrn5bw6.mongodb.net/?appName=10PearlsAQICluster
# client = MongoClient(MONGO_URI, server_api=ServerApi('1'))
# db = client["aqi_project"]
# collection = db["features"]

# df = pd.DataFrame(list(collection.find())).drop(columns=["_id"])

# st.subheader("Latest AQI Data")
# st.dataframe(df.tail(10))

# def forecast_72_hours(model, last_row):
#     future_predictions = []
#     current = last_row.copy()

#     for _ in range(72):
#         pred = model.predict(current)[0]
#         future_predictions.append(pred)

#         # Update rolling values
#         current["aqi"] = pred
#         current["pm25_change"] = pred - current["aqi"]
#         current["hour"] = (current["hour"] + 1) % 24

#     return future_predictions

# # -----------------------------
# # Prediction
# # -----------------------------
# features = [
#     "aqi",
#     "pm10",
#     "hour",
#     "day",
#     "month",
#     "day_of_week",
#     "pm25_change",
#     "pm10_change"
# ]

# latest = df[features].tail(1)
# future_aqi = forecast_72_hours(model, latest)

# forecast_df = pd.DataFrame({
#     "hour": range(1, 73),
#     "predicted_aqi": future_aqi
# })
# st.subheader("72-Hour AQI Forecast")
# st.line_chart(forecast_df.set_index("hour"))


# # latest = df[features].tail(1)

# # prediction = model.predict(latest)[0]

# # st.metric("Predicted Next Hour PM2.5", round(prediction, 2))

# # -----------------------------
# # Chart
# # -----------------------------
# st.subheader("PM2.5 Trend")
# st.line_chart(df.set_index("date")["pm2_5"].tail(200))


# Updated frontend

import streamlit as st
import pandas as pd
import numpy as np
import io
import os
import joblib
import gridfs

from pymongo import MongoClient
from pymongo.server_api import ServerApi

# ----------------------------------
# Page Config
# ----------------------------------
st.set_page_config(page_title="AQI Prediction", layout="centered")
st.title("🌫 AQI Prediction Dashboard - Karachi")

# ----------------------------------
# MongoDB Connection (USE ENV VAR)
# ----------------------------------
# MONGO_URI = "mongodb+srv://rizwan4601146_db_user:5G1bLMeOqUeWviQP@10pearlsaqicluster.rrn5bw6.mongodb.net/?appName=10PearlsAQICluster"
MONGO_URI = os.getenv("MONGO_URI")

client = MongoClient(MONGO_URI, server_api=ServerApi("1"))
db = client["aqi_project"]
collection = db["features"]
registry = db["model_registry"]
fs = gridfs.GridFS(db)

# ----------------------------------
# Load Best Model from Registry
# ----------------------------------
best_model_doc = registry.find_one({"is_best": True})

if best_model_doc is None:
    st.error("No model found in registry.")
    st.stop()

model_bytes = fs.get(best_model_doc["model_file_id"]).read()
model = joblib.load(io.BytesIO(model_bytes))

st.success(f"Using Best Model: {best_model_doc['model_name']}")
st.write(f"R² Score: {round(best_model_doc['r2'], 4)}")

# ----------------------------------
# Load Feature Data
# ----------------------------------
df = pd.DataFrame(list(collection.find())).drop(columns=["_id"])

st.subheader("Latest AQI Data")
st.dataframe(df.tail(5))

# ----------------------------------
# AQI Category Function
# ----------------------------------
def get_aqi_category(aqi):
    if aqi <= 50:
        return "Good", "green"
    elif aqi <= 100:
        return "Moderate", "orange"
    elif aqi <= 150:
        return "Unhealthy for Sensitive", "red"
    elif aqi <= 200:
        return "Unhealthy", "purple"
    else:
        return "Very Unhealthy", "maroon"

# ----------------------------------
# 72-Hour Forecast Function
# ----------------------------------
def forecast_72_hours(model, last_row):
    future_predictions = []
    current = last_row.copy()

    for _ in range(72):
        pred = model.predict(current)[0]
        future_predictions.append(pred)

        # Update rolling AQI
        current["aqi"] = pred
        current["hour"] = (current["hour"] + 1) % 24
    
    # Calculate average AQI over 72 hours
    average_aqi = sum(future_predictions) / len(future_predictions)
    print(f"Average AQI over next 72 hours: {average_aqi:.2f}")

    return future_predictions, average_aqi

# ----------------------------------
# Prepare Features
# ----------------------------------
features = [
    "aqi",
    "pm10",
    "hour",
    "day",
    "month",
    "day_of_week",
    "pm25_change",
    "pm10_change"
]

latest = df[features].tail(1)

# ----------------------------------
# Current AQI
# ----------------------------------
current_aqi = latest["aqi"].values[0]
category, color = get_aqi_category(current_aqi)

st.subheader("Current AQI Status")
st.metric("Current AQI", round(current_aqi, 2))
st.markdown(f"<h3 style='color:{color};'>{category}</h3>", unsafe_allow_html=True)

# Alert
if current_aqi > 150:
    st.error("⚠ Hazardous Air Quality Detected!")
elif current_aqi > 100:
    st.warning("⚠ Air Quality is Unhealthy for Sensitive Groups")

# ----------------------------------
# 72-Hour Forecast
# ----------------------------------
future_aqi, average_aqi = forecast_72_hours(model, latest)

# Display average AQI
avg_category, avg_color = get_aqi_category(average_aqi)
st.subheader("Average AQI Over Next 72 Hours")
st.metric("Average AQI", round(average_aqi, 2))
st.markdown(f"<h3 style='color:{avg_color};'>{avg_category}</h3>", unsafe_allow_html=True)

# Display alerts based on average AQI
if average_aqi > 150:
    st.error("⚠ Hazardous Air Quality Expected in Next 72 Hours!")
elif average_aqi > 100:
    st.warning("⚠ Air Quality Unhealthy for Sensitive Groups in Next 72 Hours")

# Hourly Forecast Table and Chart
forecast_df = pd.DataFrame({
    "Hour Ahead": range(1, 73),
    "Predicted AQI": future_aqi
})

st.subheader("72-Hour AQI Forecast (Hourly)")
st.line_chart(forecast_df.set_index("Hour Ahead"))

# ----------------------------------
# Historical Trend
# ----------------------------------
st.subheader("Recent AQI Trend (Last 200 Hours)")
st.line_chart(df.set_index("date")["aqi"].tail(200))
