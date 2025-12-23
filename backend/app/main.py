from fastapi import FastAPI
from contextlib import asynccontextmanager

from backend.app.core.config import logger
from backend.app.services.ml_service import ml_service
from backend.app.api.routers import health, predict, screen

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

@app.get("/")
async def root():
    return {"message": "Welcome to the Drug Screening System API"}

app.include_router(health.router, tags=["Health"])
app.include_router(predict.router, tags=["Prediction"])
app.include_router(screen.router, tags=["Screening"])

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
