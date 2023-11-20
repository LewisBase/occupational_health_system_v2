import numpy as np 

from diagnose_info.auditory_diagnose import AuditoryDiagnose
from detection_info.auditory_detection import PTAResult

if __name__ == "__main__":
    data = {"L-500": 20, "L-1000": 20, "L-2000": 20, "L-3000": 20, "L-4000": 20, "L-6000": 20, "L-8000": np.nan,
            "R-500": 20, "R-1000": 30, "R-2000": 40, "R-3000": 50, "R-4000": 60, "R-6000": "yddf", "R-8000": np.nan}
    pta_test = PTAResult(data=data, mean_key=[3000,4000,6000], better_strategy="optimum")
    NIPTS_test = AuditoryDiagnose.NIPTS(detection_result=pta_test, sex="F", age=34, diagnose_strategy="left")
    print(1)