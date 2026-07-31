from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from .models.database import Base, engine
from .routers.predict import router as predict_router
from .services.ml_service import ml_service


@asynccontextmanager
async def lifespan(app: FastAPI):
    Base.metadata.create_all(bind=engine)
    try:
        ml_service.load_models()
        print("ML models loaded successfully")
    except FileNotFoundError as e:
        print(f"Warning: ML models not loaded - {e}")
    yield


app = FastAPI(
    title="Bone Scaffold Property Predictor API",
    description="AI-powered predictions for bone scaffold mechanical and biological properties",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(predict_router)


@app.get("/")
def root():
    return {
        "service": "Bone Scaffold Property Predictor API",
        "version": "1.0.0",
        "docs": "/docs",
    }
