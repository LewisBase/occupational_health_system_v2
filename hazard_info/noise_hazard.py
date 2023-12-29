import pandas as pd
import numpy as np

from pydantic import BaseModel
from functional import seq
from typing import List, Dict
from pathlib import Path
from loguru import logger
from .base_hazard import BaseHazard
from constants import AuditoryConstants


def load_data(file_path, sheet_name_prefix, usecols, col_names, header):
    if file_path.exists():
        xls = pd.ExcelFile(file_path)
        valid_sheet_names = seq(xls.sheet_names).filter(
            lambda x: x.startswith(sheet_name_prefix)).list()
        if len(valid_sheet_names) > 1:
            raise ValueError("Too many valid sheet in File!")
        if len(valid_sheet_names) == 0:
            raise ValueError("No valid sheet in File")
        sheet_name = valid_sheet_names[0]
        origin_df = pd.read_excel(file_path,
                                  usecols=usecols,
                                  names=col_names,
                                  sheet_name=sheet_name,
                                  header=header)
        useful_info = {}
        for col in origin_df.columns:
            if origin_df[col].value_counts().shape[0] == 1:
                useful_info[col] = origin_df[col].unique().tolist()[0]
            else:
                useful_info[col] = origin_df[col].tolist()
        parameters_from_file = {}
        for key in useful_info.keys():
            if "kurtosis_" in key:
                parameters_from_file[key] = useful_info.pop(key)
        useful_info["parameters_from_file"] = parameters_from_file
    else:
        raise FileNotFoundError(f"Can not find file {file_path.resolve()}!!!")
    return useful_info


class NoiseHazard(BaseHazard):
    SPL_dB: List[float] = []
    SPL_dBA: List[float] = []
    SPL_dBC: List[float] = []
    kurtosis: List[float] = []
    A_kurtosis: List[float] = []
    C_kurtosis: List[float] = []
    kurtosis_median: float = None
    kurtosis_arimean: float = None
    kurtosis_geomean: float = None
    A_kurtosis_median: float = None
    A_kurtosis_arimean: float = None
    A_kurtosis_geomean: float = None
    C_kurtosis_median: float = None
    C_kurtosis_arimean: float = None
    C_kurtosis_geomean: float = None
    Leq: float = np.nan
    LAeq: float = np.nan
    LCeq: float = np.nan
    L_adjust: Dict[str, float] = {}
    parameters_from_file: Dict[str, float] = {}

    def __init__(self, **data):
        super().__init__(**data)
        self._build(**data)

    def _build(self, **data):
        self._cal_mean_kurtosis()

    @classmethod
    def load_from_file(cls,
                       recorder: str,
                       recorder_time: str,
                       parent_path: str = ".",
                       file_name: str = "Kurtosis_Leq_60s_AC.xls",
                       **kwargs):
        file_path_default = Path(
            parent_path) / recorder_time / recorder / file_name
        file_path = kwargs.pop("file_path", file_path_default)
        sheet_name_prefix = kwargs.pop("sheet_name_prefix", "Second=60")
        usecols = kwargs.pop("usecols", [
            9, 10, 11, 21, 22, 23, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35,
            36
        ])
        col_names = kwargs.pop("col_names", [
            "kurtosis", "A_kurtosis", "C_kurtosis", "SPL_dB", "SPL_dBA",
            "SPL_dBC", "Leq", "LAeq", "LCeq", "kurtosis_median",
            "kurtosis_arimean", "kurtosis_geomean", "A_kurtosis_median",
            "A_kurtosis_arimean", "A_kurtosis_geomean", "C_kurtosis_median",
            "C_kurtosis_arimean", "C_kurtosis_geomean"
        ])
        header = kwargs.pop("header", 0)

        useful_info = load_data(file_path=file_path,
                                sheet_name_prefix=sheet_name_prefix,
                                usecols=usecols,
                                col_names=col_names,
                                header=header)
        useful_info.update({"recorder":recorder, "recorder_time":recorder_time})
        return cls(**useful_info)

    def _cal_mean_kurtosis(self):
        self.kurtosis_median = np.median(self.kurtosis)
        self.kurtosis_arimean = np.mean(self.kurtosis)
        self.kurtosis_geomean = 10**(np.mean(np.log10(self.kurtosis)))
        self.A_kurtosis_median = np.median(self.A_kurtosis)
        self.A_kurtosis_arimean = np.mean(self.A_kurtosis)
        self.A_kurtosis_geomean = 10**(np.mean(np.log10(self.A_kurtosis)))
        self.C_kurtosis_median = np.median(self.C_kurtosis)
        self.C_kurtosis_arimean = np.mean(self.C_kurtosis)
        self.C_kurtosis_geomean = 10**(np.mean(np.log10(self.C_kurtosis)))
        value_check_dict = {
            "kurtosis_median": self.kurtosis_median,
            "kurtosis_arimean": self.kurtosis_arimean,
            "kurtosis_geomean": self.kurtosis_geomean,
            "A_kurtosis_median": self.A_kurtosis_median,
            "A_kurtosis_arimean": self.A_kurtosis_arimean,
            "A_kurtosis_geomean": self.A_kurtosis_geomean,
            "C_kurtosis_median": self.C_kurtosis_median,
            "C_kurtosis_arimean": self.C_kurtosis_arimean,
            "C_kurtosis_geomean": self.C_kurtosis_geomean,
        }
        for key, value in value_check_dict.items():
            if self.parameters_from_file.get(key
            ) and abs(value - self.parameters_from_file[key]) > 1E-2:
                logger.warning(
                    f"Arimean Kurtosis {round(value,3)} does not match the \
                              value {round(self.parameters_from_file[key],3)} load from file!!!"
                )
                logger.warning("Value load from file used!!!")
                value = self.parameters_from_file[key]

    def cal_adjust_L(self,
                     Lambda: float = 6.5,
                     method: str = "total_ari",
                     algorithm_code: str = "A+n",
                     **kwargs):
        effect_SPL = kwargs.get("effect_SPL", 0)
        beta_baseline = kwargs.get("beta_baseline",
                                   AuditoryConstants.BASELINE_NOISE_KURTOSIS)

        L_code = algorithm_code.split("+")[0]
        K_code = algorithm_code.split("+")[1]
        cal_parameter = {
            "n": {
                "L": self.Leq,
                "kurtosis_arimean": self.kurtosis_arimean,
                "kurtosis_geomean": self.kurtosis_geomean,
                "kurtosis": self.kurtosis,
                "SPL": self.SPL_dB
            },
            "A": {
                "L": self.LAeq,
                "kurtosis_arimean": self.A_kurtosis_arimean,
                "kurtosis_geomean": self.A_kurtosis_geomean,
                "kurtosis": self.A_kurtosis,
                "SPL": self.SPL_dBA
            },
            "C": {
                "L": self.LCeq,
                "kurtosis_arimean": self.C_kurtosis_arimean,
                "kurtosis_geomean": self.C_kurtosis_geomean,
                "kurtosis": self.C_kurtosis,
                "SPL": self.SPL_dBC
            },
        }

        if method == "total_ari":
            res = cal_parameter[L_code]["L"] + Lambda * np.log10(
                cal_parameter[K_code]["kurtosis_arimean"] /
                beta_baseline) if cal_parameter[K_code][
                    "kurtosis_arimean"] > beta_baseline else cal_parameter[
                        L_code]["L"]
        elif method == "total_geo":
            res = cal_parameter[L_code]["L"] + Lambda * np.log10(
                cal_parameter[K_code]["kurtosis_geomean"] /
                beta_baseline) if cal_parameter[K_code][
                    "kurtosis_geomean"] > beta_baseline else cal_parameter[
                        L_code]["L"]
        elif method == "segment_ari":
            if len(cal_parameter[K_code]["kurtosis"]) != len(
                    cal_parameter[L_code]["SPL"]):
                raise ValueError("kurtosis data length != SPL data length!")
            adjust_SPL_dBAs = []
            for i in range(len(cal_parameter[K_code]["kurtosis"])):
                if cal_parameter[L_code]["SPL"][i] >= effect_SPL:
                    adjust_SPL_dBA = cal_parameter[L_code]["SPL"][
                        i] + Lambda * np.log10(
                            cal_parameter[K_code]["kurtosis"][i] /
                            beta_baseline) if cal_parameter[K_code]["kurtosis"][
                                i] > beta_baseline else cal_parameter[L_code][
                                    "SPL"][i]
                else:
                    adjust_SPL_dBA = cal_parameter[L_code]["SPL"][i]
                adjust_SPL_dBAs.append(adjust_SPL_dBA)

            res = 10 * np.log10(np.mean(10**(np.array(adjust_SPL_dBAs) / 10)))
        elif method == "segment_geo":
            if len(cal_parameter[K_code]["kurtosis"]) != len(
                    cal_parameter[L_code]["SPL"]):
                raise ValueError(
                    "kurtosis data length != SPL data length!")
            adjust_SPL_dBAs = []
            for i in range(len(cal_parameter[K_code]["kurtosis"])):
                if cal_parameter[L_code]["SPL"][i] >= effect_SPL:
                    adjust_SPL_dBA = cal_parameter[L_code]["SPL"][
                        i] + Lambda * np.log10(
                            cal_parameter[K_code]["kurtosis"][i] /
                            beta_baseline)
                else:
                    adjust_SPL_dBA = cal_parameter[L_code]["SPL"][i]
                adjust_SPL_dBAs.append(adjust_SPL_dBA)

            res = np.mean(adjust_SPL_dBAs)

        self.L_adjust[method] = {"value": res, "algorithm": algorithm_code}
