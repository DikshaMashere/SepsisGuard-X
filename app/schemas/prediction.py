from pydantic import BaseModel, ConfigDict, Field


class PredictionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    Hour: float = Field(ge=0, description="Hours since ICU time-series start")
    HR: float = Field(gt=0, le=300, description="Heart rate")
    O2Sat: float = Field(gt=0, le=100, description="Oxygen saturation")
    SBP: float = Field(gt=0, le=300, description="Systolic blood pressure")
    MAP: float = Field(gt=0, le=250, description="Mean arterial pressure")
    Resp: float = Field(gt=0, le=100, description="Respiratory rate")
    Age: float = Field(ge=0, le=120, description="Age in years")
    Gender: float = Field(ge=0, le=1, description="Dataset gender encoding")
    HospAdmTime: float = Field(description="Hours between hospital admission and ICU")
    ICULOS: float = Field(ge=0, description="ICU length of stay in hours")
