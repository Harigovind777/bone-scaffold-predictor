from sqlalchemy import Column, Integer, String, Float, DateTime, JSON
from sqlalchemy.sql import func
from .database import Base


class Prediction(Base):
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    biopolymer_type = Column(String, nullable=False)
    pore_size = Column(Float, nullable=False)
    porosity = Column(Float, nullable=False)
    printing_method = Column(String, nullable=False)
    layer_thickness = Column(Float, nullable=False)
    material_composition = Column(Float, nullable=False)
    density = Column(Float, nullable=False)

    model_used = Column(String, nullable=False)
    compressive_strength = Column(Float)
    elastic_modulus = Column(Float)
    degradation_rate = Column(Float)
    cell_viability = Column(Float)
    bone_regeneration_score = Column(Float)
    input_data = Column(JSON)
