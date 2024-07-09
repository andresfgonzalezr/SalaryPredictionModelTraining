from pydantic import BaseModel
from typing import Optional


class InputData(BaseModel):
    age: Optional[str]
    industry: Optional[str]
    currency: Optional[str]
    country: Optional[str]
    us_state: Optional[str]
    city: Optional[str]
    years_experience_overall: Optional[str]
    years_experience_field: Optional[str]
    education_level: Optional[str]
    gender: Optional[str]
    race: Optional[str]
