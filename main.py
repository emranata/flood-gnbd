from datetime import date, timedelta

import joblib
import pandas as pd
import requests
from fastapi import FastAPI, HTTPException

# ---- Model (saved bundle from the notebook) ----
bundle = joblib.load("gonbad_flood_random_forest.joblib")
model = bundle["model"]
feature_columns = bundle["feature_columns"]
alert_threshold = bundle["alert_threshold"]

# ---- Settings ----
LAT, LON = 37.25, 55.17          # Gonbad-e Kavus
WET_DAY_MM = 1.0                 # a day counts as "wet" at or above this rainfall (verify against your CSV)
HISTORY_DAYS = 30                # the longest rolling window the model uses

# Open-Meteo daily variable -> column name used when the model was trained
DAILY_VARS = {
    "precipitation_sum": "precipitation_mm",
    "precipitation_hours": "precipitation_hours",
    "temperature_2m_max": "temp_max_c",
    "temperature_2m_min": "temp_min_c",
    "temperature_2m_mean": "temp_mean_c",
    "relative_humidity_2m_mean": "relative_humidity_mean_pct",
    "wind_speed_10m_max": "wind_speed_max_kmh",
    "wind_gusts_10m_max": "wind_gusts_max_kmh",
    "et0_fao_evapotranspiration": "et0_fao_mm",
    "soil_moisture_0_to_7cm_mean": "soil_moisture_0_7cm",
}

app = FastAPI(title="Gonbad-e Kavus flood model")


def fetch_daily(target: date) -> pd.DataFrame:
    """Download the last 30 days (ending on `target`) from Open-Meteo."""
    start = target - timedelta(days=HISTORY_DAYS - 1)
    # The forecast API covers about the last 92 days; older dates come from the archive.
    recent = start >= date.today() - timedelta(days=90)
    url = (
        "https://api.open-meteo.com/v1/forecast"
        if recent
        else "https://archive-api.open-meteo.com/v1/archive"
    )
    params = {
        "latitude": LAT,
        "longitude": LON,
        "start_date": start.isoformat(),
        "end_date": target.isoformat(),
        "daily": ",".join(DAILY_VARS),
        "timezone": "auto",
    }
    try:
        r = requests.get(url, params=params, timeout=30)
        r.raise_for_status()
        daily = r.json()["daily"]
    except Exception as e:
        raise HTTPException(502, f"Could not get data from Open-Meteo: {e}")

    df = pd.DataFrame(daily).rename(columns=DAILY_VARS)
    df = df.rename(columns={"time": "date"})
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date").reset_index(drop=True)


def build_features(df: pd.DataFrame, target: date) -> dict:
    """Turn 30 days of weather into the 20 inputs the model expects."""
    if len(df) < HISTORY_DAYS:
        raise HTTPException(502, f"Expected {HISTORY_DAYS} days of data, got {len(df)}.")

    p = df["precipitation_mm"]
    last = df.iloc[-1]
    feats = {col: float(last[col]) for col in DAILY_VARS.values()}
    feats.update(
        precip_sum_3d_mm=float(p.tail(3).sum()),
        precip_sum_7d_mm=float(p.tail(7).sum()),
        precip_sum_14d_mm=float(p.tail(14).sum()),
        precip_sum_30d_mm=float(p.tail(30).sum()),
        wet_days_7d=float((p.tail(7) >= WET_DAY_MM).sum()),
        wet_days_14d=float((p.tail(14) >= WET_DAY_MM).sum()),
        precipitation_lag_1d_mm=float(p.iloc[-2]),
        precipitation_lag_2d_mm=float(p.iloc[-3]),
        month=float(target.month),
        day_of_year=float(target.timetuple().tm_yday),
    )
    missing = [c for c in feature_columns if c not in feats or pd.isna(feats[c])]
    if missing:
        raise HTTPException(502, f"Open-Meteo returned no value for: {missing}")
    return feats


@app.get("/predict")
def predict(day: str | None = None):
    """Flood score for Gonbad-e Kavus. `day` is optional (YYYY-MM-DD); default is today."""
    try:
        target = date.fromisoformat(day) if day else date.today()
    except ValueError:
        raise HTTPException(400, "Use the date format YYYY-MM-DD.")
    if target > date.today():
        raise HTTPException(400, "Future dates are not supported; this model scores a day from its own weather.")

    feats = build_features(fetch_daily(target), target)
    X = pd.DataFrame([{c: feats[c] for c in feature_columns}])
    score = float(model.predict_proba(X)[0, 1])
    return {
        "location": "Gonbad-e Kavus, Golestan, Iran",
        "date": target.isoformat(),
        "model_score": round(score, 3),
        "above_alert_threshold": score >= alert_threshold,
        "alert_threshold": alert_threshold,
        "inputs_used": {k: round(v, 3) for k, v in feats.items()},
        "note": "Score is not a calibrated probability. Trained on one flood event; classifies same-day flooding, not a multi-day forecast.",
    }
