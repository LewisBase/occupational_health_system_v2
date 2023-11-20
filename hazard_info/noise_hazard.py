import pandas as pd
import numpy as np

from pydantic import BaseModel
from functional import seq
from typing import List, Dict
from pathlib import Path
from loguru import logger
from .base_hazard import BaseHazard
from constants import AuditoryConstants


class NoiseHazard(BaseHazard):
    kurtosis: List[float] = []
    kurtosis_arimean: float = None
    kurtosis_geomean: float = None
    SPL_dBA: List[float] = []
    LAeq: float = np.nan
    LAeq_adjust: Dict[str, float] = {}
    parameters_from_file: Dict[str, float] = {}

    def __init__(self, **data):
        super().__init__(**data)
        self._build(**data)

    def _build(self, **data):
        # Lambda = data.get("Lambda", 6.5)
        if self.kurtosis == [] or self.SPL_dBA == [] or np.isnan(self.LAeq):
            self._load_from_file(**data)
            self._cal_mean_kurtosis()
        if len(self.kurtosis) > 1 and not any(np.isnan(self.kurtosis)):
            self._cal_mean_kurtosis()
        # if self.kurtosis_arimean is None or self.kurtosis_geomean is None:
        #     self._cal_mean_kurtosis()

    def _load_from_file(self, **kwargs):
        parent_path = kwargs.get("parent_path", ".")
        file_name = kwargs.get("file_name", "Kurtosis_Leq.xls")
        file_name_alternative = kwargs.get("file_name", "Kurtosis_Leq-old.xls")
        usecols = kwargs.get("usecols", [9, 20, 23, 25, 26])
        col_names = kwargs.get("col_names", [
            "kurtosis", "SPL_dBA", "LAeq", "kurtosis_arimean",
            "kurtosis_geomean"
        ])
        header = kwargs.get("header", 0)

        def load_data(file_path, usecols, col_names, header=header):
            if file_path.exists():
                xls = pd.ExcelFile(file_path)
                valid_sheet_names = seq(xls.sheet_names).filter(
                    lambda x: x.startswith("Win_width=40")).list()
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
                self.kurtosis = origin_df["kurtosis"].tolist()
                self.SPL_dBA = origin_df["SPL_dBA"].tolist()
                self.LAeq = origin_df["LAeq"].unique().tolist()[0]
                self.parameters_from_file["kurtosis_arimean"] = origin_df[
                    "kurtosis_arimean"].unique().tolist()[0]
                self.parameters_from_file["kurtosis_geomean"] = origin_df[
                    "kurtosis_geomean"].unique().tolist()[0]
            else:
                raise FileNotFoundError(
                    f"Can not find file {file_path.resolve()}!!!")

        try:
            file_path = Path(
                parent_path) / self.recorder_time / self.recorder / file_name
            logger.info(f"Try to Load Noise file {file_path}")
            load_data(file_path=file_path,
                      usecols=usecols,
                      col_names=col_names)
        except FileNotFoundError:
            file_path = Path(
                parent_path
            ) / self.recorder_time / self.recorder / file_name_alternative
            logger.info(f"Try to load Noise file {file_path}")
            load_data(file_path=file_path,
                      usecols=usecols,
                      col_names=col_names)

    def _cal_mean_kurtosis(self):
        # 这里有坑，汇总数据和明细数据对不上时需要决定用哪边的数据
        # self.kurtosis_arimean = np.mean(self.kurtosis)
        if self.kurtosis_arimean is None:
            self.kurtosis_arimean = np.mean(self.kurtosis)
        self.kurtosis_geomean = 10**(np.mean(np.log10(self.kurtosis)))
        # if abs(self.kurtosis_arimean - self.parameters_from_file["kurtosis_arimean"]) > 1E-3:
        #     logger.error(f"Arimean Kurtosis {round(self.kurtosis_arimean,3)} does not match the \
        #                   value {round(self.parameters_from_file['kurtosis_arimean'],3)} load from file!!!")
        # if abs(self.kurtosis_geomean - self.parameters_from_file["kurtosis_geomean"]) > 1E-3:
        #     logger.error(f"Geomean Kurtosis {round(self.kurtosis_geomean,3)} does not match the \
        #                   value {round(self.parameters_from_file['kurtosis_geomean'],3)} load from file!!!")

    def cal_adjust_LAeq(self,
                        Lambda: float = 6.5,
                        method: str = "total_ari",
                        **kwargs):
        effect_SPL = kwargs.get("effect_SPL", 0)
        beta_baseline = kwargs.get("beta_baseline",
                                   AuditoryConstants.BASELINE_NOISE_KURTOSIS)

        if method == "total_ari":
            res = self.LAeq + Lambda * np.log10(
                self.kurtosis_arimean / beta_baseline
            ) if self.kurtosis_arimean > beta_baseline else self.LAeq
        elif method == "total_geo":
            res = self.LAeq + Lambda * np.log10(
                self.kurtosis_geomean / beta_baseline
            ) if self.kurtosis_geomean > beta_baseline else self.LAeq
        elif method == "segment_ari":
            if len(self.kurtosis) != len(self.SPL_dBA):
                raise ValueError(
                    "kurtosis data length != SPL_dBA data length!")
            adjust_SPL_dBAs = []
            for i in range(len(self.kurtosis)):
                if self.SPL_dBA[i] >= effect_SPL:
                    adjust_SPL_dBA = self.SPL_dBA[i] + Lambda * np.log10(
                        self.kurtosis[i] / beta_baseline
                    ) if self.kurtosis[i] > beta_baseline else self.SPL_dBA[i]
                else:
                    adjust_SPL_dBA = self.SPL_dBA[i]
                adjust_SPL_dBAs.append(adjust_SPL_dBA)

            res = 10 * np.log10(np.mean(10**(np.array(adjust_SPL_dBAs) / 10)))
            # res = 10 * np.log10(np.mean(10**((np.array(self.SPL_dBA) + Lambda * np.log10(
            #     np.array(self.kurtosis) / AuditoryConstants.GAUSSIAN_NOISE_KURTOSIS))/10)))
        elif method == "segment_geo":
            if len(self.kurtosis) != len(self.SPL_dBA):
                raise ValueError(
                    "kurtosis data length != SPL_dBA data length!")
            adjust_SPL_dBAs = []
            for i in range(len(self.kurtosis)):
                if self.SPL_dBA[i] >= effect_SPL:
                    adjust_SPL_dBA = self.SPL_dBA[i] + Lambda * np.log10(
                        self.kurtosis[i] / beta_baseline)
                else:
                    adjust_SPL_dBA = self.SPL_dBA[i]
                adjust_SPL_dBAs.append(adjust_SPL_dBA)

            res = np.mean(adjust_SPL_dBAs)
            # res = np.mean(np.log10(10**(np.array(self.SPL_dBA) + Lambda * np.log10(
            #     np.array(self.kurtosis) / AuditoryConstants.GAUSSIAN_NOISE_KURTOSIS))))

        self.LAeq_adjust[method] = res
