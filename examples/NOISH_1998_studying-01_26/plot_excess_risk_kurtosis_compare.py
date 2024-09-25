# -*- coding: utf-8 -*-
"""
@DATE: 2024-06-12 11:25:14
@Author: Liu Hengjiang
@File: examples\\NOISH_1998_studying-01_26\plot_excess_risk_kurtosis_compare.py
@Software: vscode
@Description:
        对比高低峰度组在不同NIHL定义及人群分组下Excess Risk的变化曲线
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
                                        key_point_xs: list,
                                        pictures_path: Path,
                                        picture_name: str = "Fig 4",
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
    task_name = kwargs.pop("task_name", "1234")

    if duration[0] == 1:
        duration_desp = "= 1~4"
    elif duration[1] == 1:
        duration_desp = "= 5~10"
    elif duration[2] == 1:
        duration_desp = "> 10"
    else:
        raise ValueError

    fig, ax = plt.subplots(1, figsize=(6.5, 5), dpi=dpi)
    for best_params_estimated, best_L_control, max_LAeq, control_params_estimated, label, color in zip(
            best_params_estimateds, best_L_controls, max_LAeqs,
            control_params_estimateds, ["KG-1", "KG-2", "KG-3"],
        ["#1f77b4", "#ff7f0e", "#2ca02c"]):
        LAeq_duration_matrix = np.tile(duration, (len(LAeq), 1)) * (
            (LAeq - best_L_control) /
            (max_LAeq - best_L_control))[:, np.newaxis]
        age_matrix = age * np.ones(len(LAeq))

        plot_X = np.concatenate(
            (age_matrix[:, np.newaxis], LAeq_duration_matrix), axis=1)
        pred_y = logistic_func_original(x=plot_X, params=best_params_estimated)
        f_prime = np.gradient(pred_y, LAeq)
        f_prime_double = np.gradient(f_prime, LAeq)
        point_x = LAeq[np.nanargmax(f_prime_double)]

        if len(control_params_estimated) == 2:
            control_X = np.array([[age]])
            control_y = logistic_func_control_0(
                x=control_X, params=control_params_estimated)
        else:
            control_X = np.array([np.concatenate([[age], duration])])
            control_y = logistic_func_control_1(
                x=control_X, params=control_params_estimated)
        # plot excess risk curve
        ax.plot(LAeq, (pred_y - control_y) * 100, label=label, color=color)
        # annotate key points
        x_min, x_max = ax.get_xlim()
        if y_lim:
            y_min, y_max = y_lim
        else:
            y_min, y_max = ax.get_ylim()
        for key_point_x in key_point_xs:
            LAeq_index = np.where(LAeq == key_point_x)[0]
            key_point_y = ((pred_y[LAeq_index] - control_y) * 100)[0]
            logger.info(
                f"{label} group excess risk at {key_point_x} = {key_point_y}")

            ax.vlines(x=key_point_x,
                      ymin=y_min,
                      ymax=key_point_y,
                      colors="black",
                      linestyles=":")
            ax.annotate("{:.2f}".format(key_point_y),
                        xy=(key_point_x, key_point_y),
                        xytext=(key_point_x - ((x_max - x_min) / 10)*int(label[-1]),
                                key_point_y + ((y_max - y_min) / 20)*int(label[-1])),
                        color=color,
                        arrowprops=dict(color=color,
                                        arrowstyle="->",
                                        linestyle="--"))

        # plot 2nd derivative key point
        # age_array = np.array([age])
        # point_x_duration_array = (point_x - best_L_control) / (
        #     max_LAeq - best_L_control) * duration
        # point_X = np.concatenate((age_array, point_x_duration_array),
        #                          axis=0)[np.newaxis, :]
        # point_y = logistic_func_original(x=point_X,
        #                                  params=best_params_estimated)
        # ax.annotate(f"key point: {point_x} dBA",
        #             xy=(point_x, (point_y - control_y) * 100),
        #             xytext=(point_x - (max(LAeq) - min(LAeq)) / 5 -
        #                     2 * int(label[-1]), (point_y - control_y) * 100 +
        #                     (y_max - y_min) / 10 + 5 * int(label[-1])),
        #             color="red",
        #             arrowprops=dict(color="red", arrowstyle="->"))

        ax.set_title(f"Age = {age}, Duration {duration_desp}")
        ax.set_ylabel("Excess Risk of HL$_{1234}$ (%)" if task_name ==
                      "1234w" else "Excess Risk of HL$_{346}$ (%)")
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
    parser.add_argument("--task_name", type=str, default="346")
    # parser.add_argument("--task_name", type=str, default="1234w")
    args = parser.parse_args()

    logger.info("Input Parameters informations:")
    message = "\n".join([f"{k:<20}: {v}" for k, v in vars(args).items()])
    logger.info(message)

    models_path = Path(args.models_path)
    pictures_path = Path(args.pictures_path)
    task_name = args.task_name

    KG1_params_estimated, KG1_L_control, KG1_max_LAeq, KG1_log_likelihood_value = pickle.load(
        open(
            models_path / Path(
                f"NIHL{task_name}_Y-KG-1-Chinese_experiment_group_udlr_model_average_freq.pkl"
            ), "rb"))
    KG2_params_estimated, KG2_L_control, KG2_max_LAeq, KG2_log_likelihood_value = pickle.load(
        open(
            models_path / Path(
                f"NIHL{task_name}_Y-KG-2-Chinese_experiment_group_udlr_model_average_freq.pkl"
            ), "rb"))
    KG3_params_estimated, KG3_L_control, KG3_max_LAeq, KG3_log_likelihood_value = pickle.load(
        open(
            models_path / Path(
                f"NIHL{task_name}_Y-KG-3-Chinese_experiment_group_udlr_model_average_freq.pkl"
            ), "rb"))

    control_params_estimated, control_log_likelihood_value = pickle.load(
        open(
            models_path /
            Path(f"HL{task_name}_Y-NOISH_control_group_udlr_model_0.pkl"),
            "rb"))

    num_res = userdefine_logistic_regression_plot(
        best_params_estimateds=[
            KG1_params_estimated, KG2_params_estimated, KG3_params_estimated
        ],
        best_L_controls=[KG1_L_control, KG2_L_control, KG3_L_control],
        max_LAeqs=[KG1_max_LAeq, KG2_max_LAeq, KG3_max_LAeq],
        control_params_estimateds=[control_params_estimated] * 3,
        key_point_xs=[80, 85, 90, 95, 100],
        pictures_path=pictures_path,
        picture_name="Fig4A" if task_name == "1234w" else "Fig4B",
        picture_format="tiff",
        age=65,
        LAeq=np.arange(60, 101),
        duration=np.array([0, 0, 1]),
        annotations={"A":
                     (-0.1,
                      1.05)} if task_name == "1234w" else {"B": (-0.1, 1.05)},
        y_lim=[-2, 60],
        task_name=task_name)

    print(1)
