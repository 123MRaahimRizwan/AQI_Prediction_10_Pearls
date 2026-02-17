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
import io
import os
import joblib
import gridfs
from pymongo import MongoClient
from pymongo.server_api import ServerApi
from dotenv import load_dotenv

load_dotenv()

# ----------------------------------
# Page Config
# ----------------------------------
st.set_page_config(page_title="AQI Prediction", layout="wide")
st.title("🌫 AQI Prediction Dashboard - Karachi")

# ----------------------------------
# MongoDB Connection
# ----------------------------------
MONGO_URI = os.getenv("MONGO_URI")

if not MONGO_URI:
    st.error("MONGO_URI environment variable not set.")
    st.stop()

client = MongoClient(MONGO_URI, server_api=ServerApi("1"))
db = client["aqi_project"]
collection = db["features"]
registry = db["model_registry"]
fs = gridfs.GridFS(db)

# ----------------------------------
# Load Best Model
# ----------------------------------
best_model_doc = registry.find_one({"is_best": True})

if not best_model_doc:
    st.error("No best model found in registry.")
    st.stop()

model_bytes = fs.get(best_model_doc["model_file_id"]).read()
model = joblib.load(io.BytesIO(model_bytes))

st.success(f"Using Best Model: {best_model_doc['model_name']}")
st.caption(f"R² Score: {round(best_model_doc['r2'], 4)}")

# ----------------------------------
# Load Feature Data
# ----------------------------------
df = pd.DataFrame(list(collection.find()))

if df.empty:
    st.warning("No feature data available yet.")
    st.stop()

df.drop(columns=["_id"], inplace=True)

st.subheader("Latest AQI Records")
st.dataframe(df.tail(5), use_container_width=True)

# ----------------------------------
# AQI Category Helper
# ----------------------------------
def get_aqi_category(aqi):
    if aqi <= 50:
        return "Good", "green"
    elif aqi <= 100:
        return "Moderate", "orange"
    elif aqi <= 150:
        return "Unhealthy for Sensitive Groups", "red"
    elif aqi <= 200:
        return "Unhealthy", "purple"
    else:
        return "Very Unhealthy", "maroon"

# ----------------------------------
# Forecast Function (72 Hours)
# ----------------------------------
def forecast_72_hours(model, last_row):
    preds = []
    current = last_row.copy()

    for _ in range(72):
        pred = model.predict(current)[0]
        preds.append(pred)

        # Update only rolling features
        current["aqi"] = pred
        current["hour"] = (current["hour"] + 1) % 24

    avg_aqi = sum(preds) / len(preds)
    return preds, avg_aqi

# ----------------------------------
# Feature Columns
# ----------------------------------
features = [
    "aqi",
    "pm10",
    "hour",
    "day",
    "month",
    "day_of_week",
    "pm25_change",
    "pm10_change",
]

latest = df[features].tail(1)

# ----------------------------------
# Current AQI
# ----------------------------------
current_aqi = float(latest["aqi"].values[0])
category, color = get_aqi_category(current_aqi)

st.subheader("Current AQI Status")
st.metric("Current AQI", round(current_aqi, 2))
st.markdown(f"<h3 style='color:{color};'>{category}</h3>", unsafe_allow_html=True)

if current_aqi > 150:
    st.error("⚠ Hazardous air quality detected!")
elif current_aqi > 100:
    st.warning("⚠ Unhealthy air quality for sensitive groups.")

# ----------------------------------
# 72-Hour Forecast
# ----------------------------------
future_aqi, avg_aqi = forecast_72_hours(model, latest)

avg_category, avg_color = get_aqi_category(avg_aqi)


future_pm25 = [aqi * 0.5 for aqi in future_aqi]

st.subheader("Average AQI (Next 72 Hours)")
st.metric("Average AQI", round(avg_aqi, 2))
st.markdown(f"<h3 style='color:{avg_color};'>{avg_category}</h3>", unsafe_allow_html=True)

if avg_aqi > 150:
    st.error("⚠ Hazardous air quality expected in next 3 days.")
elif avg_aqi > 100:
    st.warning("⚠ Unhealthy air quality expected for sensitive groups.")

# ----------------------------------
# Forecast Chart
# ----------------------------------
forecast_df = pd.DataFrame({
    "Hour Ahead": range(1, 73),
    "AQI": future_aqi,
    "PM2.5": future_pm25
})

forecast_df = forecast_df.set_index("Hour Ahead")
st.subheader("72-Hour AQI Forecast")
st.line_chart(forecast_df, use_container_width=True)

# 3 day summary

# ----------------------------------
# 3-Day Summary Cards
# ----------------------------------

forecast_df["date"] = pd.date_range(
    start=pd.Timestamp.now(),
    periods=72,
    freq="H"
)

summary = forecast_df.groupby(
    forecast_df["date"].dt.date
)["AQI"].agg(["mean", "max", "min"]).reset_index()

st.subheader("3-Day Summary")

cols = st.columns(len(summary))

for i, row in summary.iterrows():
    avg_aqi = row["mean"]
    max_aqi = row["max"]
    min_aqi = row["min"]

    category, color = get_aqi_category(avg_aqi)

    with cols[i]:
        st.markdown(
            f"""
            <div style='
                background-color:#1e2430;
                padding:20px;
                border-radius:15px;
                text-align:center;
            '>
                <h3 style="font-size: 18px;">{row['date']}</h3>
                <p style="text-align: center;"><b>Avg AQI:</b> {avg_aqi:.2f}</p>
                <p style="text-align: center;"><b>Max:</b> {max_aqi:.2f}</p>
                <p style="text-align: center;"><b>Min:</b> {min_aqi:.2f}</p>
                <h4 style='color:{color}; text-align: center;'>{category}</h4>
            </div>
            """,
            unsafe_allow_html=True
        )


# ----------------------------------
# Historical Trend
# ----------------------------------
st.subheader("AQI Trend (Last 200 Hours)")
st.line_chart(df.set_index("date")["aqi"].tail(200))

