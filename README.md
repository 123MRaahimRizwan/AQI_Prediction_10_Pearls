# Karachi AQI Forecasting System

An end-to-end **Machine Learning + MLOps pipeline** for forecasting Air Quality Index (AQI) in Karachi using automated feature engineering, model training, GitHub Actions CI/CD, MongoDB model registry, and a live Streamlit dashboard.

---

## 🚀 Project Overview

This project predicts the **next 72 hours of AQI** for Karachi by:

- Fetching air quality data from Open-Meteo API
- Engineering time-series features
- Training multiple ML models
- Storing models in a MongoDB model registry (GridFS)
- Automatically retraining via GitHub Actions
- Serving predictions via a live Streamlit dashboard

The system is fully automated and production-ready.

---

## 🏗 System Architecture

```
Open-Meteo API
      ↓
Feature Pipeline (Hourly - GitHub Actions)
      ↓
MongoDB Atlas (Features Collection)
      ↓
Training Pipeline (Daily - GitHub Actions)
      ↓
Model Registry (GridFS + Metrics)
      ↓
Streamlit Cloud Dashboard
```

---

## ⚙️ Tech Stack

### Backend / ML
- Python 3.9
- Pandas
- NumPy
- Scikit-learn
- SHAP
- Joblib

### Data & Storage
- MongoDB Atlas
- GridFS (Model Storage)

### Automation
- GitHub Actions (CI/CD)

### Frontend
- Streamlit Cloud

### API
- Open-Meteo Air Quality API

---

## 📊 Features

### 🔹 Automated Data Pipeline
- Hourly data ingestion
- Upsert logic (no duplication)
- 3-month rolling backfill
- MongoDB storage

### 🔹 Model Training Pipeline
- Multiple models comparison
- RMSE, MAE, R² evaluation
- Automatic best-model selection
- Model versioning in registry

### 🔹 Model Registry
Stores:
- Model file (GridFS)
- R²
- RMSE
- MAE
- Timestamp
- `is_best` flag

### 🔹 Live Dashboard
- Current AQI status with health category
- 72-hour forecast (AQI + PM2.5)
- 3-day summary cards (Min, Max, Avg)
- Production model metrics
- SHAP feature importance visualization
- Historical AQI trend

---

## 📁 Project Structure

```
aqi-project/
│
├── app.py # Streamlit dashboard
├── backfill.py # Historical data ingestion
├── feature_pipeline.py # Hourly feature update
├── train_pipeline.py # Daily model training
│
├── requirements.txt
│
├── .github/
│ └── workflows/
│ ├── hourly_pipeline.yml
│ └── daily_training.yml
│
└── README.md
```


---

## 🔄 Automation (GitHub Actions)

### ⏰ Hourly Pipeline
- Fetches latest AQI data
- Updates MongoDB using upsert logic

### 🌙 Daily Training Pipeline
- Trains models
- Evaluates metrics
- Updates model registry
- Marks best model as production

Runs automatically — no local machine required.

---

## 🔐 Environment Variables

The following environment variable must be set:

```MONGO_URI=your_mongodb_atlas_connection_string```


### Where to Configure
- GitHub → Repository → Settings → Secrets
- Streamlit Cloud → App Settings → Secrets

---

## 🌐 Deployment

### Backend Automation
GitHub Actions (runs even when PC is off)

### Frontend Deployment
Streamlit Cloud:
1. Connect GitHub repo
2. Select `app.py`
3. Add `MONGO_URI` in Secrets
4. Deploy

---

## 📈 Model Performance (Example)

```
      | Metric | Value |
      |--------|-------|
      | R²     | 0.90+ |
      | RMSE   | ~9    |
      | MAE    | ~5.8  |
```
Performance varies based on retraining cycle.

---

## 🧠 Feature Engineering

- Time features (hour, day, month, weekday)
- Rolling statistics
- Lag features
- PM2.5 change rates
- AQI conversion from PM2.5

---

## 🛡 Production Considerations

- No hardcoded credentials
- Automated retraining
- Model registry version control
- Cloud deployment
- Stateless frontend
- Scalable architecture

---

## 📌 Future Improvements

- Dockerization
- CI testing before model promotion
- Drift detection
- Real-time alert system
- Multi-city forecasting
- API endpoint for predictions

---

## 👨‍💻 Author

Muhammad Raahim Rizwan
Data Science Undergrad @ NED University of Engineering and Technology

---

## 📜 License

This project is for educational and portfolio purposes.
