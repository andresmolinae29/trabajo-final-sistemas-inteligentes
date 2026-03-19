from pydantic import BaseModel, Field, ConfigDict
from typing import Any


class DetectionResult(BaseModel):
    label: str
    confidence: float | Any
    bbox: tuple[float, float, float, float]


class VideoResponse(BaseModel):
    reasoning: str = Field(alias="razonamiento")
    result: str = Field(alias="resultado")
    confidence_percentage: float = Field(alias="confianza_porcentaje")
    video_name: str | None = Field(default=None, alias="nombre_video")
    
    model_config = ConfigDict(populate_by_name=True)