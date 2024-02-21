# -*- coding: utf-8 -*-
"""
@DATE: 2024-02-19 16:43:25
@Author: Liu Hengjiang
@File: examples\\NOISH_1998_studying-01_26\extract_all_Chinese_data.py
@Software: vscode
@Description:
        提取所有中国工人的噪声暴露数据
"""

import re
import pickle
import pandas as pd
import numpy as np
import pickle
from joblib import Parallel, delayed
from pathlib import Path
from functional import seq
from itertools import product
from loguru import logger

from staff_info import StaffInfo
from diagnose_info.auditory_diagnose import AuditoryDiagnose

general_calculate_func = {
    "arimean": np.mean,
    "median": np.median,
    "geomean": lambda x: 10**(np.mean(np.log10(x))),
}


def _extract_data_for_task(data: StaffInfo, **additional_set):
    mean_key = additional_set.pop("mean_key")
    better_ear_strategy = additional_set.pop("better_ear_strategy")
    NIPTS_diagnose_strategy = additional_set.pop("NIPTS_diagnose_strategy")

    res = {}
    res["staff_id"] = data.staff_id
    # worker information
    res["name"] = data.staff_basic_info.name
    res["factory"] = data.staff_id.split("-")[0]
    res["sex"] = data.staff_basic_info.sex
    res["age"] = data.staff_basic_info.age
    res["duration"] = data.staff_basic_info.duration
    res["work_shop"] = data.staff_basic_info.work_shop
    res["work_position"] = data.staff_basic_info.work_position
    res["smoking"] = data.staff_basic_info.smoking
    res["year_of_smoking"] = data.staff_basic_info.year_of_smoking
    PTA_res = data.staff_health_info.auditory_detection.get("PTA")
    res.update(seq(PTA_res.data.items()).map(
        lambda x: (x[0], float(x[1]) if re.fullmatch(r"-?\d+(\.\d+)?", str(x[1])) else np.nan)).dict())
    # label information
    res["NIPTS"] = data.staff_health_info.auditory_diagnose.get("NIPTS")
    res["NIPTS_pred_2013"] = data.NIPTS_predict_iso1999_2013(percentrage=50, mean_key=mean_key)
    res["NIPTS_pred_2023"] = data.NIPTS_predict_iso1999_2023(percentrage=50, mean_key=mean_key)
    # for freq in [1000, 2000, 3000, 4000, 6000]:
    #     res["NIPTS_" + str(freq)] = AuditoryDiagnose.NIPTS(
    #         detection_result=data.staff_health_info.auditory_detection["PTA"],
    #         sex=data.staff_basic_info.sex,
    #         age=data.staff_basic_info.age,
    #         mean_key=[freq],
    #         NIPTS_diagnose_strategy=NIPTS_diagnose_strategy)
    # feature information
    ## L
    res["Leq"] = data.staff_occupational_hazard_info.noise_hazard_info.Leq
    res["LAeq"] = data.staff_occupational_hazard_info.noise_hazard_info.LAeq
    res["LCeq"] = data.staff_occupational_hazard_info.noise_hazard_info.LCeq
    ## adjust L
    for method, algorithm_code in product(
        ["total_ari", "total_geo", "segment_ari"],
        ["A+n", "A+A", "C+n", "C+C"]):
        res[f"L{algorithm_code[0]}eq_adjust_{method}_{algorithm_code}"] = data.staff_occupational_hazard_info.noise_hazard_info.L_adjust[
            method].get(algorithm_code)
    ## kurtosis
    res["kurtosis_arimean"] = data.staff_occupational_hazard_info.noise_hazard_info.kurtosis_arimean
    res["A_kurtosis_arimean"] = data.staff_occupational_hazard_info.noise_hazard_info.A_kurtosis_arimean
    res["C_kurtosis_arimean"] = data.staff_occupational_hazard_info.noise_hazard_info.C_kurtosis_arimean
    res["kurtosis_geomean"] = data.staff_occupational_hazard_info.noise_hazard_info.kurtosis_geomean
    res["A_kurtosis_geomean"] = data.staff_occupational_hazard_info.noise_hazard_info.A_kurtosis_geomean
    res["C_kurtosis_geomean"] = data.staff_occupational_hazard_info.noise_hazard_info.C_kurtosis_geomean
    ## Peak SPL
    res["max_Peak_SPL_dB"] = data.staff_occupational_hazard_info.noise_hazard_info.Max_Peak_SPL_dB
    ## other features in frequency domain
    # for key, value in data.staff_occupational_hazard_info.noise_hazard_info.parameters_from_file.items(
    # ):
    #     if (re.findall(r"\d+",
    #                    key.split("_")[1])
    #             if len(key.split("_")) > 1 else False):
    #         if key.split("_")[0] != "Leq":
    #             for func_name, func in general_calculate_func.items():
    #                 res[key + "_" + func_name] = func(value)
    #         else:
    #             res[key] = value
    ## recorder message
    res["recorder"] = data.staff_occupational_hazard_info.noise_hazard_info.recorder
    res["recorder_time"] = data.staff_occupational_hazard_info.noise_hazard_info.recorder_time

    return res


def extract_data_for_task(df, n_jobs=-1, **additional_set):
    res = Parallel(n_jobs=n_jobs, backend="multiprocessing")(
        delayed(_extract_data_for_task)(data=data, **additional_set)
        for data in df)
    res_df = pd.DataFrame(res)
    return res_df



if __name__ == "__main__":
    from datetime import datetime
    logger.add(f"./log/extract_all_Chinese_data-{datetime.now().strftime('%Y-%m-%d')}.log",level="INFO")
    
    
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path",
                        type=str,
                        default="./cache/extract_Chinese_data.pkl")
    parser.add_argument("--output_path", type=str, default="./results")
    parser.add_argument("--picture_path", type=str, default="./pictures")
    parser.add_argument("--model_path", type=str, default="./models")
    parser.add_argument("--additional_set",
                        type=dict,
                        default={
                            "mean_key": [3000, 4000, 6000],
                            "better_ear_strategy": "optimum_freq",
                            "NIPTS_diagnose_strategy": "better"
                        })
    parser.add_argument("--n_jobs", type=int, default=-1)
    args = parser.parse_args()

    logger.info("Input Parameters informations:")
    message = "\n".join([f"{k:<20}: {v}" for k, v in vars(args).items()])
    logger.info(message)

    input_path = Path(args.input_path)
    output_path = Path(args.output_path)
    picture_path = Path(args.picture_path)
    model_path = Path(args.model_path)
    additional_set = args.additional_set
    n_jobs = args.n_jobs

    for output in (output_path, picture_path, model_path):
        if not output.exists():
            output.mkdir(parents=True)

    original_data = pickle.load(open(input_path, "rb"))
    original_data = seq(original_data).flatten().list()

    extract_df = extract_data_for_task(df=original_data,
                                       n_jobs=n_jobs,
                                       **additional_set)
    extract_df.to_csv(output_path / "all_Chinese_extract_df.csv",
                      header=True,
                      index=False)

    print(1)