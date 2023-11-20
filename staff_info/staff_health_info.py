import pandas as pd

from pydantic import BaseModel
from typing import Union, Dict
from collections import defaultdict

from constants.global_constants import AuditoryNametoObject



class StaffHealthInfo(BaseModel):
    staff_id: str
    sex: str
    age: Union[int, float]
    diagnoise_type: dict = defaultdict(lambda: [])
    auditory_detection: Dict = None
    auditory_diagnose: Dict = {}
    health_info: Dict[str, Union[str,float]] = None
    
    def __init__(self, **data):
        super().__init__(**data)
        self._build(**data)
        self._diagnoise(**data)
        
    def _build(self, **data):
        if self.auditory_detection:
            self.diagnoise_type["auditory"] = list(self.auditory_detection.keys())
            for key, value in self.auditory_detection.items():
                self.auditory_detection[key] = AuditoryNametoObject.DETECTION_TYPE_DICT[key](data=value, **data)
    
    def _diagnoise(self, **data):
        for key, value in self.auditory_detection.items():
            func = AuditoryNametoObject.DIAGNOSE_TYPE_DICT[key]
            func_name = func.__name__.split("(")[0]
            # TODO Need to calculate multi func simulataneusly
            self.auditory_diagnose[func_name] = func(detection_result=value, **data)
        