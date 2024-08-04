# -*- coding: utf-8 -*-
"""
@DATE: 2024-06-11 15:39:49
@Author: Liu Hengjiang
@File: examples\\NOISH_1998_studying-01_26\plot_excess_risk_compare.py
@Software: vscode
@Description:
        对比不同NIHL定义及人群分组下Excess Risk的变化曲线
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


def userdefine_logistic_regression_plot(best_params_estimateds: list,
                                        best_L_controls: list,
                                        max_LAeqs: list,
                                        control_params_estimateds: list,
                                        pictures_path: Path,
                                        picture_name: str = "Fig4",
                                        picture_format: str = "tiff",
                                        age: int = 30,
                                        LAeq: np.array = np.arange(70, 100),
                                        duration: np.array = np.array(
                                            [1, 0, 0]),
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
    for best_params_estimated, best_L_control, max_LAeq, control_params_estimated, label in zip(
            best_params_estimateds, best_L_controls, max_LAeqs,
            control_params_estimateds,
        ["$\\text{NIHL}_{1234}$", "$\\text{NIHL}_{346}$"]):
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

        ax.plot(LAeq, (pred_y - control_y)*100, label=label)
        ax.set_title(f"Age = {age}, Duration {duration_desp}")
        ax.set_ylabel("Excess Risk of NIHL (%)")
        ax.set_xlabel("$L_{Aeq,8h}$ (dBA)")
    plt.legend(loc="upper left")
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
            Path("NIHL1234_Y-Chinese_experiment_group_udlr_model.pkl"), "rb"))
    experiment_params_estimated_346, experiment_L_control_346, experiment_max_LAeq_346, experiment_log_likelihood_value_346 = pickle.load(
        open(
            models_path /
            Path("NIHL346_Y-Chinese_experiment_group_udlr_model.pkl"), "rb"))

    control_params_estimated_1234, control_log_likelihood_value_1234 = pickle.load(
        open(
            models_path /
            Path("NIHL1234_Y-Chinese_control_group_udlr_model_0.pkl"), "rb"))
    control_params_estimated_346, control_log_likelihood_value_346 = pickle.load(
        open(
            models_path /
            Path("NIHL346_Y-Chinese_control_group_udlr_model_0.pkl"), "rb"))

    num_res = userdefine_logistic_regression_plot(best_params_estimateds=[experiment_params_estimated_1234, experiment_params_estimated_346],
                                                  best_L_controls=[experiment_L_control_1234, experiment_L_control_346],
                                                  max_LAeqs=[experiment_max_LAeq_1234, experiment_max_LAeq_346],
                                                  control_params_estimateds=[control_params_estimated_1234, control_params_estimated_346],
                                                  pictures_path=pictures_path,
                                                  picture_name="Fig4A",
                                                  picture_format="tiff",
                                                  age=30,
                                                  LAeq=np.arange(60, 120),
                                                  duration=np.array([1,0,0]),
                                                  annotations={"A": (-0.1, 1.05)},
                                                  y_lim=[0,100])
    num_res = userdefine_logistic_regression_plot(best_params_estimateds=[experiment_params_estimated_1234, experiment_params_estimated_346],
                                                  best_L_controls=[experiment_L_control_1234, experiment_L_control_346],
                                                  max_LAeqs=[experiment_max_LAeq_1234, experiment_max_LAeq_346],
                                                  control_params_estimateds=[control_params_estimated_1234, control_params_estimated_346],
                                                  pictures_path=pictures_path,
                                                  picture_name="Fig4B",
                                                  picture_format="tiff",
                                                  age=45,
                                                  LAeq=np.arange(60, 120),
                                                  duration=np.array([0,1,0]),
                                                  annotations={"B": (-0.1, 1.05)},
                                                  y_lim=[0,100])
    num_res = userdefine_logistic_regression_plot(best_params_estimateds=[experiment_params_estimated_1234, experiment_params_estimated_346],
                                                  best_L_controls=[experiment_L_control_1234, experiment_L_control_346],
                                                  max_LAeqs=[experiment_max_LAeq_1234, experiment_max_LAeq_346],
                                                  control_params_estimateds=[control_params_estimated_1234, control_params_estimated_346],
                                                  pictures_path=pictures_path,
                                                  picture_name="Fig4C",
                                                  picture_format="tiff",
                                                  age=45,
                                                  LAeq=np.arange(60, 120),
                                                  duration=np.array([0,0,1]),
                                                  annotations={"C": (-0.1, 1.05)},
                                                  y_lim=[0,100])
    num_res = userdefine_logistic_regression_plot(best_params_estimateds=[experiment_params_estimated_1234, experiment_params_estimated_346],
                                                  best_L_controls=[experiment_L_control_1234, experiment_L_control_346],
                                                  max_LAeqs=[experiment_max_LAeq_1234, experiment_max_LAeq_346],
                                                  control_params_estimateds=[control_params_estimated_1234, control_params_estimated_346],
                                                  pictures_path=pictures_path,
                                                  picture_name="Fig4D",
                                                  picture_format="tiff",
                                                  age=65,
                                                  LAeq=np.arange(60, 120),
                                                  duration=np.array([0,0,1]),
                                                  annotations={"D": (-0.1, 1.05)},
                                                  y_lim=[0,100])

    print(1)
