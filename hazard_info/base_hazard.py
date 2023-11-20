from pydantic import BaseModel
from typing import List, Dict

class BaseHazard(BaseModel):
    recorder: str = None
    recorder_time: str = None

    def __init__(self, **data):
        super().__init__(**data)
        self._build(**data)
        
    def _build(self, **data):
        pass