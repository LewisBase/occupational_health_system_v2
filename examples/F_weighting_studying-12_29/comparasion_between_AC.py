# -*- coding: utf-8 -*-
"""
@DATE: 2024-01-02 11:15:29
@Author: Liu Hengjiang
@File: examples\F_weighting_studying-12_29\comparasion_between_AC.py
@Software: vscode
@Description:
        对比AC计权在NIPTS上的表现
"""

import re
import ast
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from joblib import Parallel, delayed
from pathlib import Path
from functional import seq
from loguru import logger

from staff_info import StaffInfo
from diagnose_info.auditory_diagnose import AuditoryDiagnose

from matplotlib.font_manager import FontProperties
from matplotlib import rcParams

config = {
    "font.family": "serif",
    "font.size": 12,
    "mathtext.fontset": "stix",  # matplotlib渲染数学字体时使用的字体，和Times New Roman差别不大
    "font.serif": ["STZhongsong"],  # 华文中宋
    "axes.unicode_minus": False  # 处理负号，即-号
}
rcParams.update(config)


def statsmodels_OLS_fit(df_box, regression_X_col, regression_y_col):
    tasks = [f"{regression_y_col} ~ {regression_X_col}"]
    models = []
    for task in tasks:
        model = sm.formula.ols(task, data=df_box).fit()
        logger.info(f"{model.summary()}")
        models.append(model)
    return models


def _extract_data_for_task(data: StaffInfo, **additional_set):
    better_ear_strategy = additional_set.pop("better_ear_strategy")
    NIPTS_diagnose_strategy = additional_set.pop("NIPTS_diagnose_strategy")
    
    res = {}
    res["staff_id"] = data.staff_id
    res["NIPTS"] = data.staff_health_info.auditory_diagnose.get("NIPTS")
    for freq in [1000, 2000, 3000, 4000, 6000]:
        res["NIPTS_"+str(freq)] = AuditoryDiagnose.NIPTS(detection_result=data.staff_health_info.auditory_detection["PTA"],
                                                         sex=data.staff_basic_info.sex,
                                                         age=data.staff_basic_info.age,
                                                         mean_key=[freq],
                                                         NIPTS_diagnose_strategy=NIPTS_diagnose_strategy)
    res["Leq"] = data.staff_occupational_hazard_info.noise_hazard_info.Leq
    res["LAeq"] = data.staff_occupational_hazard_info.noise_hazard_info.LAeq
    res["LCeq"] = data.staff_occupational_hazard_info.noise_hazard_info.LCeq
    res["kurtosis_geomean"] = data.staff_occupational_hazard_info.noise_hazard_info.kurtosis_geomean
    res["A_kurtosis_geomean"] = data.staff_occupational_hazard_info.noise_hazard_info.A_kurtosis_geomean
    res["C_kurtosis_geomean"] = data.staff_occupational_hazard_info.noise_hazard_info.C_kurtosis_geomean
    return res


def extract_data_for_task(df, n_jobs=-1, **additional_set):
    res = Parallel(n_jobs=n_jobs, backend="multiprocessing")(
        delayed(_extract_data_for_task)(data=data, **additional_set)
        for data in df)
    res_df = pd.DataFrame(res)
    return res_df


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path",
                        type=str,
                        default="./cache/extract_data.pkl")
    parser.add_argument("--output_path", type=str, default="./results")
    parser.add_argument("--additional_set",
                        type=dict,
                        default={
                            "mean_key": [3000, 4000, 6000],
                            "better_ear_strategy": "optimum",
                            "NIPTS_diagnose_strategy": "better"
                        })
    parser.add_argument("--n_jobs", type=int, default=-1)
    args = parser.parse_args()

    logger.info("Input Parameters informations:")
    message = "\n".join([f"{k:<20}: {v}" for k, v in vars(args).items()])
    logger.info(message)

    input_path = Path(args.input_path)
    output_path = Path(args.output_path)
    additional_set = args.additional_set
    n_jobs = args.n_jobs

    if not output_path.exists():
        output_path.mkdir(parents=True)

    original_data = pickle.load(open(input_path, "rb"))

    extract_df = extract_data_for_task(df=original_data,
                                       n_jobs=n_jobs,
                                       **additional_set)
    extract_df.index = extract_df.staff_id
    extract_df.drop("staff_id", axis=1, inplace=True)
    extract_df.to_csv(output_path/"extract_df.csv", header=True, index=True)

    # for X_col in ["Leq", "LAeq", "LCeq"]:
    #     models = statsmodels_OLS_fit(df_box=extract_df,
    #                                  regression_X_col=X_col,
    #                                  regression_y_col="NIPTS")


    print(1)
