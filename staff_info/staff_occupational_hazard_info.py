import pickle
import pandas as pd
import numpy as np

from pydantic import BaseModel
from typing import Union, Dict, List

from hazard_info import NoiseHazard


class StaffOccupationalHazardInfo(BaseModel):
    staff_id: str
    hazard_type: List[str] = []
    noise_hazard_info: Dict = None
    occupational_hazard_info: Dict[str, Union[str, float]] = None

    def __init__(self, **data):
        super().__init__(**data)
        self._build(**data)

    def _build(self, **data):
        if self.noise_hazard_info:
            self.hazard_type.append("noise")
            self.noise_hazard_info = NoiseHazard(**self.noise_hazard_info)
