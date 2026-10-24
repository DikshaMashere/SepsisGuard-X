from pydantic import BaseModel, Field, model_validator

from app.schemas.prediction import PredictionRequest


class TrendPredictionRequest(BaseModel):
    observations: list[PredictionRequest] = Field(
        min_length=2,
        max_length=24,
        description="Chronologically ordered observations for one patient",
    )

    @model_validator(mode="after")
    def validate_chronology(self):
        for previous, current in zip(
            self.observations,
            self.observations[1:],
        ):
            if current.Hour <= previous.Hour:
                raise ValueError(
                    "Hour must increase strictly between observations"
                )

            if current.ICULOS <= previous.ICULOS:
                raise ValueError(
                    "ICULOS must increase strictly between observations"
                )

        return self