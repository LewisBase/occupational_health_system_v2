# -*- coding: utf-8 -*-
"""
@DATE: 2024-07-22 10:00:56
@Author: Liu Hengjiang
@File: examples\\NOISH_1998_studying-01_26\Chinese_logistic_regression_control_data_Bootstrap.py
@Software: vscode
@Description:
        针对中国工人噪声暴露数据对照组进行Bootstrap采样后获取置信区间
"""

import pickle
import pandas as pd
import numpy as np
from tqdm import tqdm
from pathlib import Path
from functional import seq
from loguru import logger
from sklearn.utils import resample
from scipy.optimize import minimize

from Chinese_logistic_regression_control_data_0 import logistic_func, log_likelihood

import matplotlib.pyplot as plt
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


def userdefine_logistic_regression_Bootstrap_task(
    resample_df: pd.DataFrame,
    params_init: list,
    y_col_name: str = "NIHL1234_Y",
    **kwargs,
):
    """_summary_
    自定义逻辑回归函数，进行Bootstrap采样计算结果概率的置信区间

    Args:
        resample_df (pd.DataFrame): _description_
        params_init (list): _description_
        phi_range (list): _description_
        L_control (float): _description_
        y_col_name (str, optional): _description_. Defaults to "NIHL1234_Y".

    Returns:
        _type_: _description_
    """
    minimize_method = kwargs.pop(
        "minimize_method", "Nelder-Mead"
    )  #"SLSQP", #"Powell", #"L-BFGS-B", #"Nelder-Mead", #"BFGS",
    minimize_options = kwargs.pop("minimize_options", {'maxiter': 10000})
    minimize_bounds = kwargs.pop("minimize_bounds", None)

    work_df = resample_df.copy()

    y = work_df[y_col_name]
    X = work_df.drop(columns=[y_col_name])
    results = minimize(log_likelihood,
                       params_init,
                       args=(X.values, y.values),
                       method=minimize_method,
                       options=minimize_options,
                       bounds=minimize_bounds)
    params_estimated = results.x
    log_likelihood_value = results.fun

    return params_estimated, log_likelihood_value


def get_background_risk(
        base_params_estimated: list,
        ages: np.array = np.arange(30, 66),
):
    """_summary_
    计算指定年龄下的背景风险

    Args:
        base_params_estimated (list): _description_
        ages (np.array, optional): _description_. Defaults to np.arange(30, 66).

    Returns:
        _type_: _description_
    """
    background_risks = []
    for age in ages:
        control_X = np.array([[age]])
        background_risk = logistic_func(x=control_X,
                                        params=base_params_estimated)
        background_risks.append(background_risk)
    return pd.DataFrame(data=background_risks, index=ages)


def bootstrap_estimates(data: pd.DataFrame,
                        base_params_estimated: list,
                        y_col_name: str,
                        n_iterations: int = 1000,
                        ages: np.array = np.arange(30, 66)):
    """_summary_
    进行Bootstrap重采样，并保存每次重采样计算所得不同年龄下的背景风险数值

    Args:
        data (pd.DataFrame): _description_
        base_params_estimated (list): _description_
        y_col_name (str): _description_
        n_iterations (int, optional): _description_. Defaults to 1000.
        ages (np.array, optional): _description_. Defaults to np.arange(30, 66).

    Returns:
        _type_: _description_
    """
    estimates = []
    for i in tqdm(range(n_iterations)):
        resample_df = resample(data, replace=True, n_samples=data.shape[0])
        params_estimated, _ = userdefine_logistic_regression_Bootstrap_task(
            resample_df=resample_df, params_init=base_params_estimated, y_col_name=y_col_name)
        background_risk = get_background_risk(base_params_estimated=params_estimated,
                                              ages=ages)
        estimates.append(background_risk)
    estimate_res = pd.concat(estimates, axis=1)
    return estimate_res


def confidence_limit_plot(plot_df: pd.DataFrame, key_point_xs: list,
                          picture_path: Path, picture_name: str, picture_format: str, **kwargs):
    dpi = kwargs.pop("dpi", 330)
    is_show = kwargs.pop("is_show", False)
    y_lim = kwargs.pop("y_lim", None)

    fig, ax = plt.subplots(1, figsize=(6.5, 5), dpi=dpi)
    ax.plot(plot_df.index,
            plot_df.background_risk * 100,
            label="$\\text{HL}_{1234}$")
    ax.plot(plot_df.index,
            plot_df.lower_bounds * 100,
            linestyle="--",
            label="$\\text{HL}_{1234}$ lower limit")
    ax.plot(plot_df.index,
            plot_df.upper_bounds * 100,
            linestyle="-.",
            label="$\\text{HL}_{1234}$ upper limit")
    x_min, x_max = ax.get_xlim()
    if y_lim:
        y_min, y_max = y_lim
    else:
        y_min, y_max = ax.get_ylim()
    for key_point_x in key_point_xs:
        key_point_y = plot_df.loc[key_point_x]["background_risk"] * 100
        ax.vlines(x=key_point_x,
                  ymin=y_min,
                  ymax=key_point_y,
                  colors="black",
                  linestyles=":")
        ax.annotate("{:.2f}".format(key_point_y),
                    xy=(key_point_x, key_point_y),
                    xytext=(key_point_x - (x_max - x_min) / 10,
                            key_point_y + (y_max - y_min) / 20),
                    color="#1f77b4",
                    arrowprops=dict(color="#1f77b4",
                                    arrowstyle="->",
                                    linestyle="--"))
    ax.set_ylim(y_min, y_max)
    ax.set_ylabel("Background Risk of Hearing Loss (%)")
    ax.set_xlabel("Age (year)")
    # ax.set_title("Chinese+NOISH Control Data")
    ax.set_title("Chinese Data")

    plt.legend(loc="best")
    plt.tight_layout()
    picture_path = Path(picture_path) / f"{picture_name}.{picture_format}"
    plt.savefig(picture_path, format=picture_format, dpi=dpi)
    if is_show:
        plt.show()
    plt.close(fig=fig)


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    from datetime import datetime
    logger.add(
        f"./log/NOISH_logistic_regression_Bootstrap-{datetime.now().strftime('%Y-%m-%d')}.log",
        level="INFO")

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_path",
        type=str,
        default="./cache/Chinese_extract_control_df_average_freq.csv")
        # default="./cache/Chinese_NOISH_extract_control_df.csv")
        # default="./cache/Chinese_extract_control_classifier_df.csv")
    parser.add_argument("--task", type=str, default="analysis")
    parser.add_argument("--output_path", type=str, default="./cache")
    parser.add_argument("--models_path", type=str, default="./models")
    parser.add_argument("--pictures_path", type=str, default="./pictures")
    parser.add_argument("--n_jobs", type=int, default=-1)
    args = parser.parse_args()

    logger.info("Input Parameters informations:")
    message = "\n".join([f"{k:<20}: {v}" for k, v in vars(args).items()])
    logger.info(message)

    input_path = Path(args.input_path)
    output_path = Path(args.output_path)
    models_path = Path(args.models_path)
    pictures_path = Path(args.pictures_path)
    task = args.task
    n_jobs = args.n_jobs
    for out_path in (output_path, models_path, pictures_path):
        if not out_path.exists():
            out_path.mkdir(parents=True)

    if task == "analysis":
        extract_df = pd.read_csv(input_path, header=0, index_col="staff_id")

        # 使用Chinese对照组数据
        ## HL1234_Y
        # fit_df = extract_df[["age", "HL1234_Y"]]
        fit_df = extract_df[["age", "NIHL1234_Y"]]

        base_params_estimated, base_log_likelihood_value = pickle.load(
            open(
                models_path /
                Path("NIHL1234_Y-Chinese_control_group_udlr_model_0_average_freq.pkl"),
                # Path("HL1234_Y-Chinese_NOISH_control_group_udlr_model_0.pkl"),
                # Path("NIHL1234_Y-Chinese_control_group_udlr_model_0.pkl"),
                "rb"))

        ages = np.arange(15, 70)

        background_risk_value = get_background_risk(
            base_params_estimated=base_params_estimated,
            ages=ages)
        estimates = bootstrap_estimates(
            data=fit_df,
            base_params_estimated=base_params_estimated,
            # y_col_name="HL1234_Y",
            y_col_name="NIHL1234_Y",
            n_iterations=1000,
            ages=ages)

        confidence_res = pd.DataFrame()
        confidence_res["background_risk"] = background_risk_value
        confidence_res["lower_bounds"] = estimates.apply(
            lambda x: np.percentile(x, 5), axis=1)
        confidence_res["upper_bounds"] = estimates.apply(
            lambda x: np.percentile(x, 95), axis=1)

        confidence_limit_plot(plot_df=confidence_res,
                              key_point_xs=[30, 45, 65],
                              picture_path=pictures_path,
                              picture_name=f"Chinese_control_conf_limit_average_freq",
                              picture_format="png",
                              dpi=100)
        #   y_lim = [-1, 85])
        print(1)
