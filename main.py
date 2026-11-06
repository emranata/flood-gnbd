import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

bundle = joblib.load("gonbad_flood_random_forest.joblib")
model = bundle["model"]
feature_columns = bundle["feature_columns"]
alert_threshold = bundle["alert_threshold"]

app = FastAPI()

class DayData(BaseModel):
    values: dict[str, float]

@app.post("/predict")
def predict(day: DayData):
    missing = [c for c in feature_columns if c not in day.values]
    if missing:
        return {"error": f"Missing features: {missing}"}
    X = pd.DataFrame([{c: day.values[c] for c in feature_columns}])
    score = float(model.predict_proba(X)[0, 1])
    return {
        "model_score": round(score, 3),
        "above_alert_threshold": score >= alert_threshold,
    }