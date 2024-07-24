import numpy as np
from functional import seq
from pydantic import BaseModel

from constants.auditory_constants import AuditoryConstants
from detection_info.auditory_detection import PTAResult


class AuditoryDiagnose(BaseModel):
    # NIPTS: float = None
    # is_NIHL: bool = False

    def __init__(self, **data):
        super().__init__(**data)
        self._build(**data)

    def _build(self, **kwargs):
        pass

    @staticmethod
    def NIPTS(detection_result: PTAResult, # type: ignore
              sex: str, age: int,
              percentrage: int = 50,
              mean_key: list = [3000, 4000, 6000],
              NIPTS_diagnose_strategy: str = "better",
              **kwargs):
        if NIPTS_diagnose_strategy == "better":
            diagnose_ear_data = detection_result.better_ear_data
        elif NIPTS_diagnose_strategy == "left":
            diagnose_ear_data = detection_result.left_ear_data
        elif NIPTS_diagnose_strategy == "right":
            diagnose_ear_data = detection_result.right_ear_data

        sex = "Male" if sex in ("Male", "男", "M", "m", "male") else "Female"
        age = AuditoryConstants.AGE_BOXING(age=age)
        percentrage = str(percentrage) + "pr"
        standard_PTA = AuditoryConstants.STANDARD_PTA_DICT.get(sex).get(age)
        standard_PTA = seq(standard_PTA.items()).filter(lambda x: int(x[0].split(
            "Hz")[0]) in mean_key).map(lambda x: (int(x[0].split("Hz")[0]), x[1])).dict()
        standard_PTA = seq(standard_PTA.items()).map(
            lambda x: (x[0], x[1].get(percentrage))).dict()

        try:
            NIPTS = np.mean([diagnose_ear_data.get(key) -
                             standard_PTA.get(key) for key in mean_key])
        except TypeError:
            raise("Better ear data is incompleted!!!")
        return NIPTS
