from pydantic import BaseModel, Field


class PredictionResponse(BaseModel):
    probability: float = Field(ge=0, le=1)
    model_name: str
    threshold: float
    threshold_result: str
    disclaimer: str
