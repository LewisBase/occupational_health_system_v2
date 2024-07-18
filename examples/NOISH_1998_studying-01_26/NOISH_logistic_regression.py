# -*- coding: utf-8 -*-
"""
@DATE: 2024-01-29 10:26:29
@Author: Liu Hengjiang
@File: examples\\NOISH_1998_studying-01_26\\NOISH_logistic_regression.py
@Software: vscode
@Description:
        复现NOISH,1998论文中有关逻辑回归的内容
        2024.07.17 更新
"""
import pickle
import pandas as pd
import numpy as np
from pathlib import Path
from loguru import logger
from joblib import Parallel, delayed

from staff_info import StaffInfo
from utils.data_helper import mark_group_name
from Chinese_logistic_regression import \
    userdefine_logistic_regression_task, userdefine_logistic_regression_plot

from matplotlib import rcParams

config = {
    "font.family": "serif",
    "font.size": 12,
    "mathtext.fontset": "stix",  # matplotlib渲染数学字体时使用的字体，和Times New Roman差别不大
    "font.serif": ["STZhongsong"],  # 华文中宋
    "axes.unicode_minus": False  # 处理负号，即-号
}
rcParams.update(config)


def age_box(age):
    if age < 17:
        return 0
    elif 17 <= age < 28:
        return 1
    elif 28 <= age < 36:
        return 2
    elif 36 <= age < 46:
        return 3
    elif 46 <= age < 54:
        return 4
    else:
        return 5


def duration_box(duration):
    if duration < 1:
        return 0
    elif 1 <= duration < 2:
        return 1
    elif 2 <= duration < 5:
        return 2
    elif 5 <= duration < 11:
        return 3
    elif 11 <= duration < 21:
        return 4
    else:
        return 5


def _extract_data_for_task(data, **additional_set):
    res = {}
    res["staff_id"] = data.staff_id
    # worker information
    res["age"] = data.staff_basic_info.age
    res["duration"] = data.staff_basic_info.duration
    res["age_box"] = age_box(data.staff_basic_info.age)
    res["duration_box"] = duration_box(data.staff_basic_info.duration)

    # worker health infomation
    res["HL1234"] = data.staff_health_info.auditory_detection.get("PTA").mean(
        mean_key=[1000, 2000, 3000, 4000])
    res["HL5123"] = data.staff_health_info.auditory_detection.get("PTA").mean(
        mean_key=[500, 1000, 2000, 3000])
    res["HL123"] = data.staff_health_info.auditory_detection.get("PTA").mean(
        mean_key=[1000, 2000, 3000])
    res["HL512"] = data.staff_health_info.auditory_detection.get("PTA").mean(
        mean_key=[500, 1000, 2000])
    res["HL346"] = data.staff_health_info.auditory_detection.get("PTA").mean(
        mean_key=[3000, 4000, 6000])

    res["HL1234_Y"] = 0 if res["HL1234"] <= 25 else 1
    res["HL5123_Y"] = 0 if res["HL5123"] <= 25 else 1
    res["HL123_Y"] = 0 if res["HL123"] <= 25 else 1
    res["HL512_Y"] = 0 if res["HL512"] <= 25 else 1
    res["HL346_Y"] = 0 if res["HL346"] <= 25 else 1

    # noise information
    res["LAeq"] = data.staff_occupational_hazard_info.noise_hazard_info.LAeq

    return res


def extract_data_for_task(df, n_jobs=-1, **additional_set):
    res = Parallel(n_jobs=n_jobs, backend="multiprocessing")(
        delayed(_extract_data_for_task)(data=data, **additional_set)
        for data in df)
    res = pd.DataFrame(res)
    return res


if __name__ == "__main__":
    from datetime import datetime
    logger.add(
        f"./log/NOISH_logistic_regression-{datetime.now().strftime('%Y-%m-%d')}.log",
        level="INFO")

    import argparse
    parser = argparse.ArgumentParser()
    # parser.add_argument("--input_path",
    #                     type=str,
    #                     default="./cache/extract_NOISH_data-1234.pkl")
    parser.add_argument("--input_path",
                        type=str,
                        default="./cache/NOISH_extract_df.csv")
    parser.add_argument("--output_path", type=str, default="./cache")
    parser.add_argument("--models_path", type=str, default="./models")
    parser.add_argument("--additional_set",
                        type=dict,
                        default={
                            "mean_key": [1000, 2000, 3000, 4000],
                            "PTA_value_fix": False,
                            "better_ear_strategy": "average_freq",
                            "NIPTS_diagnose_strategy": "better"
                        })
    parser.add_argument("--task", type=str, default="analysis")
    # parser.add_argument("--task", type=str, default="extract")
    parser.add_argument("--n_jobs", type=int, default=-1)
    args = parser.parse_args()

    logger.info("Input Parameters informations:")
    message = "\n".join([f"{k:<20}: {v}" for k, v in vars(args).items()])
    logger.info(message)

    input_path = Path(args.input_path)
    output_path = Path(args.output_path)
    models_path = Path(args.models_path)
    additional_set = args.additional_set
    task = args.task
    n_jobs = args.n_jobs
    for out_path in (output_path, models_path):
        if not out_path.exists():
            out_path.mkdir(parents=True)

    if task == "extract":
        original_data = pickle.load(open(input_path, "rb"))
        extract_df = extract_data_for_task(df=original_data,
                                           n_jobs=n_jobs,
                                           **additional_set)
        extract_df.index = extract_df.staff_id
        extract_df.drop("staff_id", axis=1, inplace=True)
        extract_df.to_csv(output_path / "NOISH_extract_df.csv",
                          header=True,
                          index=True)
    if task == "analysis":
        extract_df = pd.read_csv(input_path, header=0, index_col="staff_id")
        duration_cut = [1, 4, 10, np.inf]
        extract_df["duration_box_best"] = extract_df["duration"].apply(
            lambda x: mark_group_name(x, qcut_set=duration_cut, prefix="D-"))
        max_LAeq = extract_df["LAeq"].max()

        ## 使用全部实验组数据
        fit_df = extract_df.query(
            "duration_box_best in ('D-1', 'D-2', 'D-3')")[[
                "age", "LAeq", "duration_box_best", "HL1234_Y"
            ]]
        best_params_estimated, best_L_control, max_LAeq, best_log_likelihood_value\
            = userdefine_logistic_regression_task(
            fit_df=fit_df,
            max_LAeq=max_LAeq,
            models_path=models_path,
            model_name="NOISH_experiment_group_udlr_model.pkl",
            y_col_name="HL1234_Y",
            params_init=[-5.05, 0.08, 2.66, 3.98, 6.42, 3.4],
            L_control_range=np.arange(55, 79))

        ## plot result
        userdefine_logistic_regression_plot(
            best_params_estimated=best_params_estimated,
            best_L_control=best_L_control,
            max_LAeq=max_LAeq,
            age=45,
            duration=np.array([0, 0, 1]),
            LAeq=np.arange(70, 100),
            point_type="2nd")

        ## plot result in paper
        userdefine_logistic_regression_plot(
            best_params_estimated=[-5.0557, 0.0812, 2.6653, 3.989, 6.4206, 3.4],
            best_L_control=73,
            max_LAeq=max_LAeq,
            age=45,
            duration=np.array([0, 0, 1]),
            LAeq=np.arange(70, 100),
            point_type="2nd")

    print(1)
