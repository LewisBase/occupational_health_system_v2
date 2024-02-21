import pandas as pd
import numpy as np
import pickle
from pydantic import BaseModel
from functional import seq

from constants import AuditoryConstants
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
        staff_id= data.pop("staff_id")
        name = data.pop("name", "")
        factory_name = data.pop("factory_name")
        work_shop = data.pop("work_shop")
        work_position = data.pop("work_position")
        sex = data.pop("sex")
        age = data.pop("age")
        duration = data.pop("duration")
        self.staff_basic_info = StaffBasicInfo(staff_id=self.staff_id,
                                               name=name,
                                               factory_name=factory_name,
                                               work_shop=work_shop,
                                               work_position=work_position,
                                               sex=sex,
                                               age=age,
                                               duration=duration,
                                               **data)

        # 员工健康诊断信息构建
        self.staff_health_info = StaffHealthInfo(staff_id=self.staff_id,
                                                 sex=self.staff_basic_info.sex,
                                                 age=self.staff_basic_info.age,
                                                 **data)

        # 员工职业危害因素信息构建
        self.staff_occupational_hazard_info = StaffOccupationalHazardInfo(staff_id=self.staff_id,
                                                                          **data)

    def NIPTS_predict_iso1999_2013(self,
                                   Hs: bool = False,
                                   percentrage: int = 50,
                                   mean_key: list = [3000, 4000, 6000],
                                   **kwargs):
        LAeq = kwargs.pop("LAeq", None) or self.staff_occupational_hazard_info.noise_hazard_info.LAeq
        age = kwargs.pop("age", None) or self.staff_basic_info.age
        sex = kwargs.pop("sex", None) or self.staff_basic_info.sex
        duration = kwargs.pop("duration", None) or self.staff_basic_info.duration
        NH_limit = kwargs.pop("NH_limit", True)
        
        sex = "Male" if sex.startswith("M") else "Female"
        age = AuditoryConstants.AGE_BOXING(age=age)
        percentrage = str(percentrage) + "pr"
        standard_PTA = AuditoryConstants.STANDARD_PTA_DICT.get(sex).get(age)
        standard_PTA = seq(standard_PTA.items()).filter(lambda x: int(x[0].split(
            "Hz")[0]) in mean_key).map(lambda x: (int(x[0].split("Hz")[0]), x[1])).dict()
        standard_PTA = seq(standard_PTA.items()).map(
            lambda x: (x[0], x[1].get(percentrage))).dict()

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
            
            if Hs:
                H = standard_PTA.get(freq)
                if NH_limit:
                    if H + NIPTS_pred > 40:
                        NIPTS_pred = NIPTS_pred - NIPTS_pred * H / 120
                else:
                    NIPTS_pred = NIPTS_pred - NIPTS_pred * H / 120
            NIPTS_preds.append(NIPTS_pred)
        NIPTS_pred_res = np.mean(NIPTS_preds)
        return NIPTS_pred_res

    def NIPTS_predict_iso1999_2023(self,
                                   percentrage: int = 50,
                                   mean_key: list = [3000, 4000, 6000],
                                   **kwargs):
        LAeq = kwargs.pop("LAeq", None) or self.staff_occupational_hazard_info.noise_hazard_info.LAeq
        age = kwargs.pop("age", None) or self.staff_basic_info.age
        sex = kwargs.pop("sex", None) or self.staff_basic_info.sex
        duration = kwargs.pop("duration", None) or self.staff_basic_info.duration
        extrapolation = kwargs.pop("extrapolation", None)
        NH_limit = kwargs.pop("NH_limit", True)

        # Calculate N (NIPTS) values
        # convert from VBA
        age = 21 if age <= 20 else age
        age = 70 if age > 70 else age
        duration = 40 if duration > 40 else duration
        duration = age - 20 if age - duration < 20 else duration
        S = "Male" if sex == "M" else "Female"

        As = [20, 30, 40, 50, 60]
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
            age = str(As[A-1])
            percentage = str(PS[P - 1]) + "pr"
            frequence = str(F) + "Hz"

            dict_1 = AuditoryConstants.ISO_1999_2023_NIPTS_PRED_DICT_B6.get(S)
            try:
                res = dict_1.get(age).get(frequence).get(percentage,0)
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

            if age < 20 or age > 70:
                NLDQ = np.nan
            if duration < 1 or duration > 40 or age - duration < 20:
                NLDQ = np.nan
            # if LAeq < 70 or LAeq > 100:
            if LAeq < 70:
                NLDQ = np.nan
            if LAeq > 100:
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
            # 仅在NH_limit开启的状况下对H + NLDQ的值进行判断后再调整，否则全部调整
            if NH_limit:
                if H + NLDQ > 40:
                    NLDQ = NLDQ - H * NLDQ / 120
            else:
                NLDQ = NLDQ - H * NLDQ / 120
            NIPTS_preds.append(NLDQ)

        NIPTS_pred_res = np.mean(NIPTS_preds)
        return NIPTS_pred_res