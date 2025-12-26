from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from app.core.config import logger
from app.services.ml_service import ml_service
from app.api.routers import health, predict, screen, system, molecule
from app.api.routers import dual_predict

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup logic
    ml_service.load_model()
    yield
    # Shutdown logic (if any)
    ml_service.predictor = None

app = FastAPI(
    title="Drug Screening System API",
    description="API for drug property prediction and screening based on Deep Learning.",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"message": "Welcome to the Drug Screening System API"}

app.include_router(health.router, tags=["Health"])
app.include_router(system.router, prefix="/system", tags=["System"])
app.include_router(predict.router, tags=["Prediction"])
app.include_router(dual_predict.router, tags=["Dual Prediction"])
app.include_router(screen.router, tags=["Screening"])
app.include_router(molecule.router, tags=["Molecule"])

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
