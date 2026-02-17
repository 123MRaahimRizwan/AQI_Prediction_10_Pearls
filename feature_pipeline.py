import openmeteo_requests
import pandas as pd
import requests_cache
from retry_requests import retry
from pymongo import MongoClient
from pymongo.server_api import ServerApi
from datetime import datetime, timedelta
import os

# -----------------------------
# MongoDB connection
# -----------------------------
# MONGO_URI = "mongodb+srv://rizwan4601146_db_user:5G1bLMeOqUeWviQP@10pearlsaqicluster.rrn5bw6.mongodb.net/?appName=10PearlsAQICluster"
MONGO_URI = os.getenv("MONGO_URI")
client = MongoClient(MONGO_URI, server_api=ServerApi("1"))
db = client["aqi_project"]
collection = db["features"]

# -----------------------------
# Get Latest Timestamp from DB
# -----------------------------
latest_doc = collection.find_one(sort=[("date", -1)])

if latest_doc:
    start_date = latest_doc["date"].date()
    print(f"Last record found: {latest_doc['date']}")
else:
    # If DB empty → fetch last 7 days
    start_date = datetime.utcnow().date() - timedelta(days=7)
    print("No existing data found. Fetching last 7 days.")

end_date = datetime.utcnow().date()

print(f"Fetching data from {start_date} to {end_date}")

# -----------------------------
# Open-Meteo Client Setup
# -----------------------------
cache_session = requests_cache.CachedSession('.cache', expire_after=3600)
retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
openmeteo = openmeteo_requests.Client(session=retry_session)

url = "https://air-quality-api.open-meteo.com/v1/air-quality"

params = {
    "latitude": 24.8607,
    "longitude": 67.0011,
    "hourly": ["pm10", "pm2_5"],
    "start_date": str(start_date),
    "end_date": str(end_date),
}

responses = openmeteo.weather_api(url, params=params)
response = responses[0]
hourly = response.Hourly()

data = {
    "date": pd.date_range(
        start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
        end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
        freq=pd.Timedelta(seconds=hourly.Interval()),
        inclusive="left",
    ),
    "pm10": hourly.Variables(0).ValuesAsNumpy(),
    "pm2_5": hourly.Variables(1).ValuesAsNumpy(),
}

df = pd.DataFrame(data)

# -----------------------------
# Feature Engineering
# -----------------------------
def pm25_to_aqi(pm25):
    breakpoints = [
        (0.0, 12.0, 0, 50),
        (12.1, 35.4, 51, 100),
        (35.5, 55.4, 101, 150),
        (55.5, 150.4, 151, 200),
        (150.5, 250.4, 201, 300),
    ]

    for pm_low, pm_high, aqi_low, aqi_high in breakpoints:
        if pm_low <= pm25 <= pm_high:
            return ((aqi_high - aqi_low) / (pm_high - pm_low)) * (pm25 - pm_low) + aqi_low

    return None

df["hour"] = df["date"].dt.hour
df["day"] = df["date"].dt.day
df["month"] = df["date"].dt.month
df["day_of_week"] = df["date"].dt.dayofweek

df["aqi"] = df["pm2_5"].apply(pm25_to_aqi)

df["pm25_change"] = df["pm2_5"].diff()
df["pm10_change"] = df["pm10"].diff()

df["target_aqi"] = df["aqi"].shift(-1)
df.dropna(inplace=True)

# -----------------------------
# Upsert into MongoDB
# -----------------------------
inserted_count = 0
updated_count = 0

for record in df.to_dict("records"):
    result = collection.update_one(
        {"date": record["date"]},
        {"$set": record},
        upsert=True
    )

    if result.upserted_id:
        inserted_count += 1
    else:
        updated_count += 1

print(f"Inserted: {inserted_count}")
print(f"Updated: {updated_count}")
print("✅ Feature pipeline completed successfully")
