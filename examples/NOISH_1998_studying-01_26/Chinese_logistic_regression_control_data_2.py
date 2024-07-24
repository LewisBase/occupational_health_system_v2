# -*- coding: utf-8 -*-
"""
@DATE: 2024-07-19 16:10:04
@Author: Liu Hengjiang
@File: examples\\NOISH_1998_studying-01_26\Chinese_logistic_regression_control_data_2.py
@Software: vscode
@Description:
        在中国工人噪声暴露数据对照组中加入NOISH数据进行尝试
        顺便对比两个对照组的差异
"""

import pickle
import pandas as pd
import numpy as np
from functional import seq
from pathlib import Path
from loguru import logger

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
    logger.add(
        f"./log/Chinese_logistic_regression_control_data_2-{datetime.now().strftime('%Y-%m-%d')}.log",
        level="INFO")

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_paths",
                        type=str,
                        default=[
                            "./cache/NOISH_extract_control_df.csv",
                            "./cache/Chinese_extract_control_classifier_df.csv"
                        ])
    parser.add_argument("--output_path", type=str, default="./cache")
    parser.add_argument("--models_path", type=str, default="./models")
    parser.add_argument("--task", type=str, default="analysis")
    args = parser.parse_args()

    logger.info("Input Parameters informations:")
    message = "\n".join([f"{k:<20}: {v}" for k, v in vars(args).items()])
    logger.info(message)

    input_paths = seq(args.input_paths).map(lambda x: Path(x))
    output_path = Path(args.output_path)
    models_path = Path(args.models_path)
    task = args.task
    for out_path in (output_path, models_path):
        if not out_path.exists():
            out_path.mkdir(parents=True)

    if task == "analysis":
        extract_df = pd.DataFrame()
        for input_path in input_paths:
            sub_extract_df = pd.read_csv(input_path,
                                         header=0,
                                         index_col="staff_id")
            sub_extract_df.columns = seq(sub_extract_df.columns).map(
                lambda x: x[2:] if x.startswith("NIHL") else x)
            drop_col = seq(sub_extract_df.columns).filter(lambda x: not x in ("age", "duration", "HL1234", "HL346", "HL1234_Y", "HL346_Y", "LAeq"))
            sub_extract_df.drop(drop_col, axis=1, inplace=True)
            sub_extract_df["Source"] = "Chinese" if sub_extract_df.shape[0] == 1626 else "NOISH"
            extract_df = pd.concat([extract_df, sub_extract_df], axis=0)
        extract_df.to_csv(output_path / "Chinese_NOISH_extract_control_df.csv",
                          header=True,
                          index=True)
        # 使用全部对照组数据
        fit_df = extract_df[["age", "HL1234_Y"]]
        best_params_estimated, best_log_likelihood_value\
            = userdefine_logistic_regression_task(
            fit_df=fit_df,
            models_path=models_path,
            model_name="Chinese_NOISH_control_group_udlr_model_0.pkl",
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
            model_name="Chinese_NOISH_control_group_udlr_model_0.pkl",
            y_col_name="HL346_Y",
            params_init=[-5.05, 0.08])

        ## plot result
        userdefine_logistic_regression_plot(
            best_params_estimated=best_params_estimated)

    print(1)
