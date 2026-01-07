from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from ..fighters import get_all_fighters, fighter_exists
from ..predict import predict_fight

app = FastAPI(title="UFC Predictor API")
app.add_middleware(CORSMiddleware, allow_origins=["*"])

class PredictionRequest(BaseModel):
    fighter1: str
    fighter2: str

class PredictionResponse(BaseModel):
    fighter1: str
    fighter2: str
    fighter1_win_probability: float
    fighter2_win_probability: float
    predicted_winner: str

@app.get("/fighters")
def list_fighters():
    return {"fighters": get_all_fighters()}

@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    if request.fighter1 == request.fighter2:
        raise HTTPException(400, "Fighter1 and Fighter2 must be different")
    
    if not fighter_exists(request.fighter1):
        raise HTTPException(404, f"Fighter '{request.fighter1}' not found")
    if not fighter_exists(request.fighter2):
        raise HTTPException(404, f"Fighter '{request.fighter2}' not found")
    
    return PredictionResponse(**predict_fight(request.fighter1, request.fighter2))
