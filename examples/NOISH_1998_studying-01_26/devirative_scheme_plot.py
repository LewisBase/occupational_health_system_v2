# -*- coding: utf-8 -*-
"""
@DATE: 2024-06-05 13:54:14
@Author: Liu Hengjiang
@File: examples\\NOISH_1998_studying-01_26\devirative_scheme_plot.py
@Software: vscode
@Description:
        概率曲线导数临界点示意图
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

from staff_info import StaffInfo
from diagnose_info.auditory_diagnose import AuditoryDiagnose
from utils.data_helper import mark_group_name, filter_data
from Chinese_control_group_logistic_regression_0 import logistic_func as logistic_func_control_0
from Chinese_control_group_logistic_regression_1 import logistic_func as logistic_func_control_1


def logistic_func(params, x):
    alpha, beta1, beta21, beta22, beta23, phi = params
    F = alpha + beta1 * x[:, 0] + beta21 * np.power(
        x[:, 1], phi) + beta22 * np.power(x[:, 2], phi) + beta23 * np.power(
            x[:, 3], phi)
    return np.exp(F) / (1 + np.exp(F))


def userdefine_logistic_regression_plot(best_params_estimated,
                                        best_L_control,
                                        max_LAeq,
                                        pictures_path: Path,
                                        picture_name: str = "Fig 1",
                                        picture_format: str = "tiff",
                                        age: int = 30,
                                        LAeq: np.array = np.arange(70, 100),
                                        duration: np.array = np.array([1, 0, 0]),
                                        point_types: list = ["max 1st", "max 2nd"],
                                        **kwargs):
    
    control_params_estimated = kwargs.pop("control_params_estimated", None)
    control_y = kwargs.pop("y_control", 0)
    dpi = kwargs.pop("dpi", 330)
    is_show = kwargs.pop("is_show", False)

    LAeq_duration_matrix = np.tile(duration, (len(LAeq), 1)) * (
        (LAeq - best_L_control) / (max_LAeq - best_L_control))[:, np.newaxis]
    age_matrix = age * np.ones(len(LAeq))

    plot_X = np.concatenate((age_matrix[:, np.newaxis], LAeq_duration_matrix),
                            axis=1)
    pred_y = logistic_func(x=plot_X, params=best_params_estimated)
    f_prime = np.gradient(pred_y, LAeq)
    f_prime_double = np.gradient(f_prime, LAeq)

    if control_params_estimated is not None:
        if len(control_params_estimated) == 2:
            control_X = np.array([[age]])
            control_y = logistic_func_control_0(
                x=control_X, params=control_params_estimated)
        else:
            control_X = np.array([np.concatenate([[age], duration])])
            control_y = logistic_func_control_1(
                x=control_X, params=control_params_estimated)

    # start plot
    fig, ax = plt.subplots(1, figsize=(6.5, 5), dpi=dpi)
    ax.plot(LAeq, (pred_y - control_y)*100, color="#1f77b4")
    ax2 = ax.twinx()
    ax2.plot(LAeq, f_prime, color="#ff7f0e", linestyle="--", label="1st derivative", alpha=0.8)
    ax2.plot(LAeq,
             f_prime_double,
             color="#2ca02c",
             linestyle="--",
             label="2nd derivative",
             alpha=0.8)
    ax2.hlines(y=0,
               xmin=min(LAeq),
               xmax=max(LAeq),
               colors="black",
               linestyles=":")
    y1_min, y1_max = ax.get_ylim()
    ## annotate special point
    for point_type in point_types:
        if point_type == "max 1st":
            point_x = LAeq[np.nanargmax(f_prime)]
            vline_ymax = max(f_prime)
        elif point_type == "max 2nd":
            point_x = LAeq[np.nanargmax(f_prime_double)]
            vline_ymax = max(f_prime_double)

        age_array = np.array([age])
        point_x_duration_array = (point_x - best_L_control) / (
            max_LAeq - best_L_control) * duration
        point_X = np.concatenate((age_array, point_x_duration_array),
                                 axis=0)[np.newaxis, :]
        point_y = logistic_func(x=point_X, params=best_params_estimated)

        ax.annotate(f"{point_type}",
                    xy=(point_x, (point_y - control_y)*100),
                    xytext=(point_x - (max(LAeq) - min(LAeq)) / 5,
                            (point_y - control_y)*100 + (y1_max - y1_min) / 10),
                    color="red",
                    arrowprops=dict(color="red", arrowstyle="->"))
        ax2.vlines(x=point_x,
                   ymin=0,
                   ymax=vline_ymax, #point_y - control_y,
                   color="black",
                   linestyles=":")
        
    ax.set_ylabel("Excess Risk of NIHL (%)")
    ax2.set_ylabel("Derivative of Excess Risk")
    ax.set_xlabel("$L_{Aeq,8h}$ (dBA)")
    
    from matplotlib.lines import Line2D
    userdefine_label={
   "Excess Risk": Line2D([], [], color="#1f77b4"),
   "1st derivative": Line2D([],[], color="#ff7f0e", alpha=0.8, linestyle="--"),
   "2nd derivative": Line2D([],[], color="#2ca02c", alpha=0.8, linestyle="--"),
   }
    # plt.Rectangle()自定义了图例的样式，hatch为填充样式
   
    ax.legend(
    		  handles=userdefine_label.values(),  # 自定义的样式
              labels=userdefine_label.keys(),     # 自定义的名称
              loc="best",                         # 位置
              fontsize="small"                    # 字体大小
              )
    plt.tight_layout()
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

    best_params_estimated, best_L_control, max_LAeq, best_log_likelihood_value = pickle.load(
        open(
            models_path /
            Path("NIHL1234_Y-Chinese_experiment_group_udlr_model.pkl"), "rb"))
    control_params_estimated_1, control_log_likelihood_value_1 = pickle.load(
        open(
            models_path /
            Path("NIHL1234_Y-Chinese_control_group_udlr_model_1.pkl"), "rb"))

    num_res = userdefine_logistic_regression_plot(
                    best_params_estimated=best_params_estimated,
                    best_L_control=best_L_control,
                    max_LAeq=max_LAeq,
                    pictures_path=pictures_path,
                    age=65,
                    LAeq=np.arange(50, 130),
                    duration=np.array([0, 1, 0]),
                    control_params_estimated=control_params_estimated_1)