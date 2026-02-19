import pandas as pd
import joblib
from pymongo import MongoClient

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge

from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score

from pymongo.server_api import ServerApi
import base64
from datetime import datetime
import io
import gridfs
import os

# -----------------------------
# MongoDB connection
# -----------------------------
MONGO_URI = os.getenv("MONGO_URI")
client = MongoClient(MONGO_URI, server_api=ServerApi('1'))

db = client["aqi_project"]
collection = db["features"]

# -----------------------------
# Load data
# -----------------------------
df = pd.DataFrame(list(collection.find()))
df.drop(columns=["_id"], inplace=True)

# -----------------------------
# Train/Test split
# -----------------------------
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
target = "target_aqi"

X = df[features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False
)

# -----------------------------
# Save model metadata
# -----------------------------
registry = db["model_registry"]
fs = gridfs.GridFS(db)

models = {
    "RandomForest": RandomForestRegressor(n_estimators=100, random_state=42),
    "LinearRegression": LinearRegression(),
    "Ridge": Ridge(alpha=1.0),
    "GradientBoosting": GradientBoostingRegressor()
}

for model_name, model in models.items():

    print(f"\nTraining {model_name}...")

    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    rmse = root_mean_squared_error(y_test, preds)
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)

    print(f"{model_name} RMSE:", rmse)
    print(f"{model_name} MAE:", mae)
    print(f"{model_name} R²:", r2)

    # Serialize model
    buffer = io.BytesIO()
    joblib.dump(model, buffer)
    model_bytes = buffer.getvalue()

    model_file_id = fs.put(
        model_bytes,
        filename=f"{model_name}_AQI_model"
    )

    registry.insert_one({
        "model_name": model_name,
        "created_at": datetime.utcnow(),
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "model_file_id": model_file_id,
        "is_best": False
    })

print("Model stored in registry")

# Best model updation
best_model = registry.find_one(sort=[("r2", -1)])

registry.update_many({}, {"$set": {"is_best": False}})

registry.update_one(
    {"_id": best_model["_id"]},
    {"$set": {"is_best": True}}
)

print(f"\nBest model selected: {best_model['model_name']}")
