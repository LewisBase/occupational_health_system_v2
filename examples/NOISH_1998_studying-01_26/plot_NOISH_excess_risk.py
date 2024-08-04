# -*- coding: utf-8 -*-
"""
@DATE: 2024-07-17 15:38:19
@Author: Liu Hengjiang
@File: examples\\NOISH_1998_studying-01_26\\plot_NOISH_excess_risk.py
@Software: vscode
@Description:
        对NOISH复现的结果进行excess risk计算并绘图
"""

import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from pathlib import Path
from functional import seq
from loguru import logger

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

from Chinese_logistic_regression import logistic_func_original as logistic_func_original
from Chinese_logistic_regression_control_data_0 import logistic_func as logistic_func_control_0
from Chinese_logistic_regression_control_data_1 import logistic_func as logistic_func_control_1


def excess_risk_cal_plot(best_params_estimated: list,
                         best_L_control: list,
                         max_LAeq: list,
                         control_params_estimated: list,
                         pictures_path: Path,
                         picture_name: str = "Fig4",
                         picture_format: str = "tiff",
                         age: int = 30,
                         LAeq: np.array = np.arange(70, 100),
                         duration: np.array = np.array([1, 0, 0]),
                         annotations={"A": (-0.1, 1.05)},
                         **kwargs):

    dpi = kwargs.pop("dpi", 330)
    is_show = kwargs.pop("is_show", False)
    y_lim = kwargs.pop("y_lim", None)

    if duration[0] == 1:
        duration_desp = "= 1~4"
    elif duration[1] == 1:
        duration_desp = "= 5~10"
    elif duration[2] == 1:
        duration_desp = "> 10"
    else:
        raise ValueError

    fig, ax = plt.subplots(1, figsize=(5, 5), dpi=dpi)
    LAeq_duration_matrix = np.tile(duration, (len(LAeq), 1)) * (
        (LAeq - best_L_control) /
        (max_LAeq - best_L_control))[:, np.newaxis]
    age_matrix = age * np.ones(len(LAeq))

    plot_X = np.concatenate(
        (age_matrix[:, np.newaxis], LAeq_duration_matrix), axis=1)
    pred_y = logistic_func_original(x=plot_X, params=best_params_estimated)

    if len(control_params_estimated) == 2:
        control_X = np.array([[age]])
        control_y = logistic_func_control_0(
            x=control_X, params=control_params_estimated)
    else:
        control_X = np.array([np.concatenate([[age], duration])])
        control_y = logistic_func_control_1(
            x=control_X, params=control_params_estimated)
    logger.info(f"control base probability: {control_y}")

    ax.plot(LAeq, (pred_y - control_y) * 100)
    ax.set_title(f"Age = {age}, Duration {duration_desp}")
    ax.set_ylabel("Excess Risk of NIHL (%)")
    ax.set_xlabel("$L_{Aeq,8h}$ (dBA)")
    # plt.legend(loc="upper left")
    for label, (x, y) in annotations.items():
        ax.annotate(label,
                    xy=(1, 0),
                    xycoords='axes fraction',
                    xytext=(x, y),
                    textcoords='axes fraction',
                    fontproperties=FontProperties(size=20, weight='bold'))
    if y_lim:
        plt.ylim(y_lim)
    picture_path = Path(pictures_path) / f"{picture_name}.{picture_format}"
    plt.savefig(picture_path, format=picture_format, dpi=dpi)
    if is_show:
        plt.show()
    plt.close(fig=fig)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--models_path", type=str, default="./models")
    parser.add_argument("--pictures_path", type=str, default="./pictures")
    args = parser.parse_args()

    logger.info("Input Parameters informations:")
    message = "\n".join([f"{k:<20}: {v}" for k, v in vars(args).items()])
    logger.info(message)

    models_path = Path(args.models_path)
    pictures_path = Path(args.pictures_path)

    experiment_params_estimated_1234, experiment_L_control_1234, experiment_max_LAeq_1234, experiment_log_likelihood_value_1234 = pickle.load(
        open(
            models_path /
            Path("HL1234_Y-NOISH_experiment_group_udlr_model.pkl"), "rb"))

    control_params_estimated_1234, control_log_likelihood_value_1234 = pickle.load(
        open(
            models_path /
            Path("HL1234_Y-NOISH_control_group_udlr_model_0.pkl"), "rb"))

    # reproduction result
    num_res = excess_risk_cal_plot(
        best_params_estimated=experiment_params_estimated_1234,
        best_L_control=experiment_L_control_1234,
        max_LAeq=experiment_max_LAeq_1234,
        control_params_estimated=control_params_estimated_1234,
        pictures_path=pictures_path,
        picture_name="NOISH_excess_risk_reproduce",
        picture_format="tiff",
        age=65,
        LAeq=np.arange(70, 100),
        duration=np.array([0, 0, 1]),
        annotations={"A": (-0.1, 1.05)},
        y_lim=[-5, 50])
    
    # paper result
    num_res = excess_risk_cal_plot(
        best_params_estimated=[-5.0557, 0.0812, 2.6653, 3.989, 6.4206, 3.4],
        best_L_control=73,
        max_LAeq=experiment_max_LAeq_1234,
        control_params_estimated=control_params_estimated_1234,
        pictures_path=pictures_path,
        picture_name="NOISH_excess_risk_paper",
        picture_format="tiff",
        age=65,
        LAeq=np.arange(70, 100),
        duration=np.array([0, 0, 1]),
        annotations={"B": (-0.1, 1.05)},
        y_lim=[-5, 50])

    print(1)
