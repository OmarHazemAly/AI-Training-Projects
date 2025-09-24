from fastapi import FastAPI
from pydantic import BaseModel, Field
import joblib
import pandas as pd
import os

app = FastAPI(title="Titanic Survival API")

MODEL_PATH = os.path.join(os.path.dirname(__file__), "titanic_model.pkl")
model = joblib.load(MODEL_PATH) if os.path.exists(MODEL_PATH) else None

class Passenger(BaseModel):
    pclass: int = Field(..., ge=1, le=3, description="Ticket class 1–3")
    sex: int = Field(..., ge=0, le=1, description="0=female, 1=male")
    age: float = Field(..., ge=0, description="Age in years")
    fare: float = Field(..., ge=0, description="Ticket fare")

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": bool(model)}

@app.post("/predict")
def predict(passenger: Passenger):
    if model is None:
        return {"error": "Model not found. Train and save titanic_model.pkl first."}
    X = pd.DataFrame([{
        "Pclass": passenger.pclass,
        "Sex_num": passenger.sex,
        "Age": passenger.age,
        "Fare": passenger.fare
    }])
    pred = int(model.predict(X)[0])
    proba = float(model.predict_proba(X)[0][1])
    return {"survived": pred, "probability": proba}
