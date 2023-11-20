import pickle
import pandas as pd
import numpy as np

from pydantic import BaseModel
from typing import Union, Dict, List

from hazard_info import NoiseHazard
from constants import AuditoryConstants


class StaffOccupationalHazardInfo(BaseModel):
    staff_id: str
    sex: str
    age: Union[int, float]
    duration: float
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

    def NIPTS_predict_iso1999_2013(self,
                                   mean_key: list = [3000, 4000, 6000],
                                   **kwargs):
        LAeq = kwargs.get("LAeq", None) or self.noise_hazard_info.LAeq
        duration = kwargs.get("duration", None) or self.duration
        NIPTS_preds = []
        for freq in mean_key:
            u = AuditoryConstants.ISO_1999_2013_NIPTS_PRED_DICT.get(
                str(freq) + "Hz").get("u")
            v = AuditoryConstants.ISO_1999_2013_NIPTS_PRED_DICT.get(
                str(freq) + "Hz").get("v")
            L0 = AuditoryConstants.ISO_1999_2013_NIPTS_PRED_DICT.get(
                str(freq) + "Hz").get("L0")
            if duration < 10:
                NIPTS_pred = np.log10(duration + 1) / np.log10(11) * (
                    u + v * np.log10(10 / 1)) * (LAeq - L0)**2
            else:
                NIPTS_pred = (u + v * np.log10(duration / 1)) * (LAeq - L0)**2
            NIPTS_preds.append(NIPTS_pred)
        NIPTS_pred_res = np.mean(NIPTS_preds)
        return NIPTS_pred_res

    def NIPTS_predict_iso1999_2023(self,
                                   percentrage: int = 50,
                                   mean_key: list = [3000, 4000, 6000],
                                   **kwargs):
        LAeq = kwargs.get("LAeq", None) or self.noise_hazard_info.LAeq
        age = kwargs.get("age", None) or self.age
        sex = kwargs.get("sex", None) or self.sex
        duration = kwargs.get("duration", None) or self.duration
        extrapolation = kwargs.get("extrapolation", None)

        # Calculate N (NIPTS) values
        # convert from VBA
        age = 21 if age <= 20 else age
        age = 70 if age > 70 else age
        duration = 40 if duration > 40 else duration
        duration = age - 20 if age - duration < 20 else duration
        S = "Male" if sex == "M" else "Female"

        A0 = (age - 10) / 10
        A1 = int(A0) if age < 70 else 5
        A2 = A1 + 1
        AR = (A0 - A1) / (A2 - A1)

        ls = [70, 75, 80, 85, 90, 95, 100]
        L0 = (LAeq - 65) / 5
        L1 = int(L0) if LAeq < 100 else 6
        L2 = L1 + 1
        LR = (L0 - L1) / (L2 - L1)

        D0 = duration / 10
        D1 = int(D0) if duration != 40 else 3
        D2 = D1 + 1
        DR = (D0 - D1) / (D2 - D1)
        D1 = D1 if duration >= 10 else 1

        ps = [90, 95, 75, 50, 25, 10, 5]
        if 90 <= percentrage <= 95:
            Q1 = 1
        elif 75 <= percentrage < 90:
            Q1 = 2
        elif 50 <= percentrage < 75:
            Q1 = 3
        elif 25 <= percentrage < 50:
            Q1 = 4
        elif 10 <= percentrage < 25:
            Q1 = 5
        elif 5 <= percentrage < 10:
            Q1 = 6
        Q2 = Q1 + 1
        QR = (percentrage - ps[Q1 - 1]) / (ps[Q2 - 1] - ps[Q1 - 1])

        def dict_query_1(L, D, P, F, PS=ps, LS=ls):
            LAeq = str(ls[L - 1]) + "dB"
            duration = str(D * 10) + "years"
            percentage = str(PS[P - 1]) + "pr"
            frequence = str(F) + "Hz"

            dict_1 = AuditoryConstants.ISO_1999_2023_NIPTS_PRED_DICT_T1.get(
                duration)
            try:
                res = dict_1.get(LAeq).get(frequence).get(percentage)
            except:
                res = np.nan
            return res

        def dict_query_2(A, S, P, F, PS=ps):
            percentage = str(PS[P - 1]) + "pr"
            frequence = str(F) + "Hz"

            dict_1 = AuditoryConstants.ISO_1999_2023_NIPTS_PRED_DICT_B6.get(S)
            try:
                res = dict_1.get(str(A)).get(frequence).get(percentage)
            except:
                res = np.nan
            return res

        NIPTS_preds = []
        for freq in mean_key:
            try:
                if duration < 10:
                    LG = int(np.log10(duration + 1) / np.log10(11) * 10 +
                             0.5) / 10
                    NQ1 = int((dict_query_1(L1, 1, Q1, freq) + LR *
                               (dict_query_1(L2, 1, Q1, freq))) * 10 +
                              0.5) / 10
                    NQ2 = int((dict_query_1(L1, 1, Q2, freq) + LR *
                               (dict_query_1(L2, 1, Q2, freq) -
                                dict_query_1(L1, 1, Q2, freq))) * 10 +
                              0.5) / 10
                    NLDQ = LG * int(((NQ1 + QR * (NQ2 - NQ1))) * 10 + 0.5) / 10
                else:
                    N1 = int((dict_query_1(L1, D1, Q1, freq) + LR *
                              (dict_query_1(L2, D1, Q1, freq) -
                               dict_query_1(L1, D1, Q1, freq))) * 10 +
                             0.5) / 10
                    N2 = int((dict_query_1(L1, D2, Q1, freq) + LR *
                              (dict_query_1(L2, D2, Q1, freq) -
                               dict_query_1(L1, D2, Q1, freq))) * 10 +
                             0.5) / 10
                    NQ1 = int(((N1 + DR * (N2 - N1))) * 10 + 0.5) / 10
                    N1 = int((dict_query_1(L1, D1, Q2, freq) + LR *
                              (dict_query_1(L2, D1, Q2, freq) -
                               dict_query_1(L1, D1, Q2, freq))) * 10 +
                             0.5) / 10
                    N2 = int((dict_query_1(L1, D2, Q2, freq) + LR *
                              (dict_query_1(L2, D2, Q2, freq) -
                               dict_query_1(L1, D2, Q2, freq))) * 10 +
                             0.5) / 10
                    NQ2 = int(((N1 + DR * (N2 - N1))) * 10 + 0.5) / 10
                    NLDQ = int((NQ1 + QR * (NQ2 - NQ1)) * 10 + 0.5) / 10
            except:
                NLDQ = np.nan
            try:
                H1 = int((dict_query_2(A1, S, Q1, freq) + AR *
                          (dict_query_2(A2, S, Q1, freq) -
                           dict_query_2(A1, S, Q1, freq))) * 10 + 0.5) / 10
                H2 = int((dict_query_2(A1, S, Q2, freq) + AR *
                          (dict_query_2(A2, S, Q2, freq) -
                           dict_query_2(A1, S, Q2, freq))) * 10 + 0.5) / 10
                N2 = int((dict_query_1(L1, D2, Q1, freq) + LR *
                          (dict_query_1(L2, D2, Q1, freq) -
                           dict_query_1(L1, D2, Q1, freq))) * 10 + 0.5) / 10
                H = int(((H1 + QR * (H2 - H1))) * 10 + 0.5) / 10
            except:
                H = np.nan

            if H + NLDQ > 40:
                NLDQ = NLDQ - H * NLDQ / 120
            if age < 20 or age > 70:
                NLDQ = np.nan
            if duration < 1 or duration > 40 or age - duration < 20:
                NLDQ = np.nan
            if LAeq < 70 or LAeq > 100:
                if extrapolation == "ML":
                    model = pickle.load(
                        open(
                            f"./model/regression_model_for_NIPTS_pred_2023.pkl",
                            "rb"))
                    feature = [[1 if sex == "M" else 0, age, duration, LAeq]]
                    NLDQ = model.predict(
                        pd.DataFrame(
                            feature,
                            columns=["sex_encoder", "age", "duration",
                                     "LAeq"]))[0]
                elif extrapolation == "Linear":
                    NIPTS_pred_res_95 = self.NIPTS_predict_iso1999_2023(
                        LAeq=95)
                    NIPTS_pred_res_100 = self.NIPTS_predict_iso1999_2023(
                        LAeq=100)
                    m = (NIPTS_pred_res_100 - NIPTS_pred_res_95) / 5
                    b = NIPTS_pred_res_100 - m * 100
                    NLDQ = m * LAeq + b
                else:
                    NLDQ = np.nan
            # if age > 70 or duration > 40 or LAeq > 100:
            # model = pickle.load(open(f"./model/regression_model_for_NIPTS_pred_2023.pkl", "rb"))
            # feature = [[1 if sex=="M" else 0, age, duration, LAeq]]
            # NLDQ = model.predict(pd.DataFrame(feature,columns=["sex_encoder","age","duration","LAeq"]))[0]
            NIPTS_preds.append(NLDQ)

        NIPTS_pred_res = np.mean(NIPTS_preds)
        return NIPTS_pred_res


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import pickle
    from loguru import logger
    from sklearn.linear_model import Lasso
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error

    from datetime import datetime
    logger.add(f"./log/staff_occupational_hazard_info-{datetime.now().strftime('%Y-%m-%d')}.log",level="INFO")

    from matplotlib.font_manager import FontProperties
    from matplotlib import rcParams
    
    config = {
                "font.family": "serif",
                "font.size": 12,
                "mathtext.fontset": "stix",# matplotlib渲染数学字体时使用的字体，和Times New Roman差别不大
                "font.serif": ["STZhongsong"],# 华文中宋
                "axes.unicode_minus": False # 处理负号，即-号
             }
    rcParams.update(config)

    # # Single debug test
    # test_res = {
    #     "staff_id": "test-1",
    #     "sex": "M",
    #     "age": "40",
    #     "duration": 10,
    #     "noise_hazard_info": {
    #         "recorder": "a",
    #         "recorder_time": "2023.11.12",
    #         "kurtosis": [1],
    #         "SPL_dBA": [1],
    #         "LAeq": 107
    #     }
    # }
    # staff_NIPTS_pred_test = StaffOccupationalHazardInfo(**test_res)
    # NIPTS_2023 = staff_NIPTS_pred_test.NIPTS_predict_iso1999_2023(
    #     extrapolation="Linear")
    # logger.info(f"test result: {NIPTS_2023}")

    # # train for meachine learning
    # total_res = {"sex":[],
    #              "age":[],
    #              "duration":[],
    #              "LAeq":[],
    #              "NIPTS":[]}
    # for sex in ("F", "M"):
    #     for age in range(20,71,1):
    #         for duration in range(1,41,1):
    #             for LAeq in range(70,100,5):
    #                 mesg_dict = {
    #                     "staff_id": "a1",
    #                     "sex": sex,
    #                     "age": age,
    #                     "duration": duration,
    #                     "noise_hazard_info": {"recorder": "a",
    #                                           "recorder_time": "2023.11.12",
    #                                           "kurtosis": [1],
    #                                           "SPL_dBA": [1],
    #                                           "LAeq": LAeq}
    #                 }
    #                 staff_occ_test = StaffOccupationalHazardInfo(**mesg_dict)
    #                 logger.info(f"sex={sex}, age={age}, duration={duration}, LAeq={LAeq}")
    #                 NIPTS_2023 = staff_occ_test.NIPTS_predict_iso1999_2023()
    #                 logger.info(f"NIPTS={NIPTS_2023}")
    #                 total_res["sex"].append(sex)
    #                 total_res["age"].append(age)
    #                 total_res["duration"].append(duration)
    #                 total_res["LAeq"].append(LAeq)
    #                 total_res["NIPTS"].append(NIPTS_2023)
    # df = pd.DataFrame(total_res)
    # plt.scatter(df["LAeq"], df["NIPTS"])
    # plt.show()
    # # train predict model
    # df["sex_encoder"] = df["sex"].map({"M":1, "F":0})
    # X = df[["sex_encoder", "age", "duration", "LAeq"]]
    # y = df["NIPTS"]
    # X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=42)
    # model = Lasso()
    # model.fit(X_train, y_train)

    # y_predict = model.predict(X_test)
    # MSE = mean_squared_error(y_pred=y_predict, y_true=y_test)
    # logger.info(f"Model performance MSE={round(MSE,3)}")
    # pickle.dump(model, open(f"./model/regression_model_for_NIPTS_pred_2023.pkl", "wb"))

    # interpolation results compare
    total_res = {"sex":[],
                 "age":[],
                 "duration":[],
                 "LAeq":[],
                 "NIPTS_2023_LI":[],
                 "NIPTS_2023_ML":[]}
    for sex in ("F", "M"):
        for age in range(20,71,1):
            for duration in range(1,41,1):
                for LAeq in range(70,200,5):
                    mesg_dict = {
                        "staff_id": "a1",
                        "sex": sex,
                        "age": age,
                        "duration": duration,
                        "noise_hazard_info": {"recorder": "a",
                                              "recorder_time": "2023.11.12",
                                              "kurtosis": [1],
                                              "SPL_dBA": [1],
                                              "LAeq": LAeq}
                    }
                    staff_occ_test = StaffOccupationalHazardInfo(**mesg_dict)
                    logger.info(f"sex={sex}, age={age}, duration={duration}, LAeq={LAeq}")
                    NIPTS_2023_LI = staff_occ_test.NIPTS_predict_iso1999_2023(extrapolation="Linear")
                    NIPTS_2023_ML = staff_occ_test.NIPTS_predict_iso1999_2023(extrapolation="ML")
                    logger.info(f"NIPTS predict with Linear={NIPTS_2023_LI}")
                    logger.info(f"NIPTS predict with ML={NIPTS_2023_ML}")
                    total_res["sex"].append(sex)
                    total_res["age"].append(age)
                    total_res["duration"].append(duration)
                    total_res["LAeq"].append(LAeq)
                    total_res["NIPTS_2023_LI"].append(NIPTS_2023_LI)
                    total_res["NIPTS_2023_ML"].append(NIPTS_2023_ML)
    df = pd.DataFrame(total_res)
    
    for col in ["NIPTS_2023_LI", "NIPTS_2023_ML"]:
        fig, ax = plt.subplots(1, figsize=(6.5,5))
        ax.scatter(df["LAeq"], df[col], alpha=0.4)
        ax.set_ylim([-10,325])
        ax.set_xlabel("$L_{Aeq}$ (dBA)")
        ax.set_ylabel("$NIPTS_{346}$ (dB)")
        ax.set_title(f"Extrapolation results by {'linear interpolation' if col == 'NIPTS_2023_LI' else 'machine learning'}")
        plt.savefig(f"./Extrapolation results by {'linear interpolation' if col == 'NIPTS_2023_LI' else 'machine learning'}.png")
        plt.show()
        plt.close(fig=fig)
        
        
    print(1)
