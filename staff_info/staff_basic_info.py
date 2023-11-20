import pandas as pd
import numpy as np

from pydantic import BaseModel
from typing import Union


class StaffBasicInfo(BaseModel):
    staff_id: str  # 工厂名称+自增id
    factory_name: str 
    work_shop: Union[str, float]
    work_position: Union[str, float]
    sex: str
    age: Union[int,float]
    duration: float # 工龄，单位年
    smoking: Union[str, int] = None
    year_of_smoking: Union[float, str] = None
    cigarette_per_day: Union[float, str] = None
    occupational_clinic_class: int = None

    def __init__(self, **data):
        super().__init__(**data)
        self._build(**data)

    def _build(self, **data):
        self.sex = "M" if self.sex in ("Male", "男", "M", "m", "male") else "F"
        if self.smoking is None or self.smoking == "N":
            self.smoking = "N"
            self.year_of_smoking = 0
            self.cigarette_per_day = 0
