from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pathlib import Path
from fighters import get_all_fighters, fighter_exists
from predict import predict_fight

app = FastAPI(title="UFC Predictor API")

# Get frontend directory path (works for both local and deployed)
frontend_dir = Path(__file__).parent.parent.parent / "frontend"

# Serve static files (CSS, JS)
if frontend_dir.exists():
    app.mount("/static", StaticFiles(directory=str(frontend_dir)), name="static")
    
    @app.get("/")
    async def read_root():
        return FileResponse(str(frontend_dir / "index.html"))

# Configure CORS (still useful for API endpoints)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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