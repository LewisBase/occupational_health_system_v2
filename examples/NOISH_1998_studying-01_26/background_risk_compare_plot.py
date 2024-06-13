# -*- coding: utf-8 -*-
"""
@DATE: 2024-06-07 14:56:17
@Author: Liu Hengjiang
@File: examples\\NOISH_1998_studying-01_26\\background_risk_compare_plot.py
@Software: vscode
@Description:
        对比不同NIHL定义下Background Risk的变化曲线
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

from Chinese_control_group_logistic_regression_0 import logistic_func


def userdefine_logistic_regression_plot(control_params_estimated_1234,
                                        control_params_estimated_346,
                                        pictures_path: Path,
                                        picture_name: str = "Fig3",
                                        picture_format: str = "tiff",
                                        age: np.array = np.arange(15, 70),
                                        key_point_xs: list = [30, 45, 65],
                                        **kwargs):
    dpi = kwargs.pop("dpi", 330)
    is_show = kwargs.pop("is_show", False)

    plot_X = age[:, np.newaxis]
    pred_y_1234 = logistic_func(x=plot_X,
                                params=control_params_estimated_1234) * 100
    pred_y_346 = logistic_func(x=plot_X,
                               params=control_params_estimated_346) * 100

    fig, ax = plt.subplots(1, figsize=(6.5, 5), dpi=dpi)
    ax.plot(plot_X, pred_y_1234, label="$\\text{NIHL}_{1234}$")
    ax.plot(plot_X, pred_y_346, label="$\\text{NIHL}_{346}$")
    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()
    for key_point_x in key_point_xs:
        key_point_y_1234 = logistic_func(
            x=np.array([[key_point_x]
                        ]), params=control_params_estimated_1234)[0] * 100
        key_point_y_346 = logistic_func(
            x=np.array([[key_point_x]
                        ]), params=control_params_estimated_346)[0] * 100
        ax.vlines(x=key_point_x,
                  ymin=y_min,
                  ymax=max(key_point_y_1234, key_point_y_346),
                  colors="black",
                  linestyles=":")
        
        ax.annotate("{:.2f}".format(key_point_y_1234),
                    xy=(key_point_x, key_point_y_1234),
                    xytext=(key_point_x - (x_max - x_min) / 5,
                            key_point_y_1234 + (y_max - y_min) / 10),
                    color="#1f77b4",
                    arrowprops=dict(color="#1f77b4", arrowstyle="->", linestyle="--"))
        ax.annotate("{:.2f}".format(key_point_y_346),
                    xy=(key_point_x, key_point_y_346),
                    xytext=(key_point_x - (x_max - x_min) / 5,
                            key_point_y_346 + (y_max - y_min) / 5),
                    color="#ff7f0e",
                    arrowprops=dict(color="#ff7f0e", arrowstyle="->", linestyle="--"))
    ax.set_ylim(y_min, y_max)
    ax.set_ylabel("Background Risk of NIHL (%)")
    ax.set_xlabel("Age (year)")
    ax.set_xticks([15,30,45,60,65,75])

    plt.legend(loc="best")
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

    control_params_estimated_1234, control_log_likelihood_value_1234 = pickle.load(
        open(
            models_path /
            Path("NIHL1234_Y-Chinese_control_group_udlr_model_0.pkl"), "rb"))
    control_params_estimated_346, control_log_likelihood_value_346 = pickle.load(
        open(
            models_path /
            Path("NIHL346_Y-Chinese_control_group_udlr_model_0.pkl"), "rb"))

    num_res = userdefine_logistic_regression_plot(
        control_params_estimated_1234=control_params_estimated_1234,
        control_params_estimated_346=control_params_estimated_346,
        pictures_path=pictures_path)

    print(1)
