import pandas as pd
import numpy as np

from pydantic import BaseModel
from .staff_basic_info import StaffBasicInfo
from .staff_health_info import StaffHealthInfo
from .staff_occupational_hazard_info import StaffOccupationalHazardInfo


class StaffInfo(BaseModel):
    staff_id: str  # 工厂名称+自增id
    staff_basic_info: StaffBasicInfo = None
    staff_health_info: StaffHealthInfo = None
    staff_occupational_hazard_info: StaffOccupationalHazardInfo = None

    def __init__(self, **data):
        super().__init__(**data)
        self._build(**data)

    def _build(self, **data):
        # 员工基础信息构建
        self.staff_basic_info = StaffBasicInfo(**data)

        # 员工健康诊断信息构建
        self.staff_health_info = StaffHealthInfo(**data)

        # 员工职业危害因素信息构建
        noise_hazard_info = data.get("noise_hazard_info", None)
        if noise_hazard_info is not None:
            self.staff_occupational_hazard_info = StaffOccupationalHazardInfo(**data)
