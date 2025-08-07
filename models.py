from pydantic import BaseModel, Field
from typing import Optional


# used Field to set constraints on the fields
# used Literal to restrict the values of Sleep_Disorder to specific options
# used Optional to allow Sleep_Disorder to be None or one of the specified values
# used ge and le to set minimum and maximum values for numeric fields
class SleepRecord(BaseModel):
    Gender: str
    Age: int
    Occupation: str
    Sleep_Duration: float = Field(..., alias="Sleep Duration")
    Quality_of_Sleep: int = Field(..., alias="Quality of Sleep")
    Physical_Activity_Level: int = Field(..., alias="Physical Activity Level")
    Stress_Level: int = Field(..., alias="Stress Level")
    BMI_Category: str = Field(..., alias="BMI Category")
    Heart_Rate: int = Field(..., alias="Heart Rate")
    Sleep_Disorder: Optional[str] = Field(None, alias="Sleep Disorder")

    class Config:
        populate_by_name = True
