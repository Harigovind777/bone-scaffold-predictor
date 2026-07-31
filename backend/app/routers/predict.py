from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from sqlalchemy import desc
from typing import Optional

from ..models.database import get_db
from ..models.schemas import PredictionInput, PredictionOutput, PredictionHistoryItem, PredictionHistoryResponse
from ..models.models import Prediction
from ..services.ml_service import ml_service

router = APIRouter(prefix="/api", tags=["predictions"])

AVAILABLE_MODELS = ["random_forest", "xgboost", "lightgbm", "neural_network"]


@router.post("/predict", response_model=PredictionOutput)
def predict(
    input_data: PredictionInput,
    model: str = Query("random_forest", description="AI model to use"),
    db: Session = Depends(get_db),
):
    if model not in AVAILABLE_MODELS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid model '{model}'. Choose from: {AVAILABLE_MODELS}"
        )

    try:
        result = ml_service.predict(input_data.model_dump(), model_name=model)
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

    prediction_record = Prediction(
        biopolymer_type=input_data.biopolymer_type,
        pore_size=input_data.pore_size,
        porosity=input_data.porosity,
        printing_method=input_data.printing_method,
        layer_thickness=input_data.layer_thickness,
        material_composition=input_data.material_composition,
        density=input_data.density,
        model_used=result["model_used"],
        compressive_strength=result["compressive_strength"],
        elastic_modulus=result["elastic_modulus"],
        degradation_rate=result["degradation_rate"],
        cell_viability=result["cell_viability"],
        bone_regeneration_score=result["bone_regeneration_score"],
        input_data=input_data.model_dump(),
    )
    db.add(prediction_record)
    db.commit()
    db.refresh(prediction_record)

    return PredictionOutput(
        compressive_strength=result["compressive_strength"],
        elastic_modulus=result["elastic_modulus"],
        degradation_rate=result["degradation_rate"],
        cell_viability=result["cell_viability"],
        bone_regeneration_score=result["bone_regeneration_score"],
        model_used=result["model_used"],
        model_confidence=result.get("model_confidence"),
    )


@router.post("/predict/all", response_model=dict)
def predict_all(input_data: PredictionInput, db: Session = Depends(get_db)):
    results = {}
    for model_name in AVAILABLE_MODELS:
        try:
            result = ml_service.predict(input_data.model_dump(), model_name=model_name)
            results[model_name] = {
                "compressive_strength": result["compressive_strength"],
                "elastic_modulus": result["elastic_modulus"],
                "degradation_rate": result["degradation_rate"],
                "cell_viability": result["cell_viability"],
                "bone_regeneration_score": result["bone_regeneration_score"],
                "confidence": result.get("model_confidence"),
            }
        except Exception as e:
            results[model_name] = {"error": str(e)}

    return {"input": input_data.model_dump(), "results": results}


@router.get("/predict/history", response_model=PredictionHistoryResponse)
def get_history(
    skip: int = Query(0, ge=0),
    limit: int = Query(20, ge=1, le=100),
    db: Session = Depends(get_db),
):
    total = db.query(Prediction).count()
    predictions = (
        db.query(Prediction)
        .order_by(desc(Prediction.created_at))
        .offset(skip)
        .limit(limit)
        .all()
    )

    items = []
    for p in predictions:
        items.append(PredictionHistoryItem(
            id=p.id,
            created_at=p.created_at,
            input_data=PredictionInput(**p.input_data),
            output_data=PredictionOutput(
                compressive_strength=p.compressive_strength,
                elastic_modulus=p.elastic_modulus,
                degradation_rate=p.degradation_rate,
                cell_viability=p.cell_viability,
                bone_regeneration_score=p.bone_regeneration_score,
                model_used=p.model_used,
            ),
        ))

    return PredictionHistoryResponse(predictions=items, total=total)


@router.get("/models")
def list_models():
    try:
        scores = ml_service.get_model_scores()
    except FileNotFoundError:
        scores = {}
    return {
        "models": [
            {
                "name": m,
                "label": m.replace("_", " ").title(),
                "r2_score": scores.get(m, {}).get("r2"),
                "rmse": scores.get(m, {}).get("rmse"),
            }
            for m in AVAILABLE_MODELS
        ]
    }
