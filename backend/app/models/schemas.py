from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime


class PredictionInput(BaseModel):
    biopolymer_type: str = Field(..., description="Type of biopolymer")
    pore_size: float = Field(..., ge=10, le=1000, description="Pore size in microns")
    porosity: float = Field(..., ge=10, le=100, description="Porosity percentage")
    printing_method: str = Field(..., description="3D printing method")
    layer_thickness: float = Field(..., ge=10, le=500, description="Layer thickness in microns")
    material_composition: float = Field(..., ge=0, le=1, description="Material composition ratio")
    density: float = Field(..., ge=0.01, le=3.0, description="Density in g/cm³")


class PredictionOutput(BaseModel):
    compressive_strength: float
    elastic_modulus: float
    degradation_rate: float
    cell_viability: float
    bone_regeneration_score: float

    model_used: str
    model_confidence: Optional[float] = None


class PredictionHistoryItem(BaseModel):
    id: int
    created_at: datetime
    input_data: PredictionInput
    output_data: PredictionOutput

    class Config:
        from_attributes = True


class PredictionHistoryResponse(BaseModel):
    predictions: List[PredictionHistoryItem]
    total: int
