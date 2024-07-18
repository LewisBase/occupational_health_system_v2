# -*- coding: utf-8 -*-
"""
@DATE: 2024-07-18 14:32:57
@Author: Liu Hengjiang
@File: examples\\NOISH_1998_studying-01_26\\NOISH_logistic_regression_control_data.py
@Software: vscode
@Description:
        针对NOISH的对照组数据进行回归拟合
"""

import pickle
import pandas as pd
import numpy as np
from pathlib import Path
from loguru import logger
from joblib import Parallel, delayed

from staff_info import StaffInfo
from utils.data_helper import mark_group_name
from NOISH_logistic_regression import age_box, duration_box, extract_data_for_task
from Chinese_logistic_regression_control_data_0 import \
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


if __name__ == "__main__":
    from datetime import datetime
    logger.add(f"./log/NOISH_logistic_regression_control_data-{datetime.now().strftime('%Y-%m-%d')}.log",level="INFO")
    
    import argparse
    parser = argparse.ArgumentParser()
    # parser.add_argument("--input_path",
    #                     type=str,
    #                     default="./cache/extract_NOISH_control_data-1234.pkl")
    parser.add_argument("--input_path",
                        type=str,
                        default="./cache/NOISH_extract_control_df.csv")
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
        extract_df.to_csv(output_path / "NOISH_extract_control_df.csv",
                          header=True,
                          index=True)
    if task == "analysis":
        extract_df = pd.read_csv(input_path, header=0, index_col="staff_id")
        # 使用全部对照组数据
        fit_df = extract_df[["age", "HL1234_Y"]]
        best_params_estimated, best_log_likelihood_value\
            = userdefine_logistic_regression_task(
            fit_df=fit_df,
            models_path=models_path,
            model_name="NOISH_control_group_udlr_model_0.pkl",
            y_col_name="HL1234_Y",
            params_init=[-5.05, 0.08])

        ## plot result
        userdefine_logistic_regression_plot(
            best_params_estimated=best_params_estimated)
        
        fit_df = extract_df[["age", "HL346_Y"]]
        best_params_estimated, best_log_likelihood_value\
            = userdefine_logistic_regression_task(
            fit_df=fit_df,
            models_path=models_path,
            model_name="NOISH_control_group_udlr_model_0.pkl",
            y_col_name="HL346_Y",
            params_init=[-5.05, 0.08])

        ## plot result
        userdefine_logistic_regression_plot(
            best_params_estimated=best_params_estimated)

    print(1)