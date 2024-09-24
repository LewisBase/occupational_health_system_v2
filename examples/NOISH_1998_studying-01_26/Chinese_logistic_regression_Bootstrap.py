# -*- coding: utf-8 -*-
"""
@DATE: 2024-06-14 17:01:35
@Author: Liu Hengjiang
@File: examples\\NOISH_1998_studying-01_26\Chinese_logistic_regression_Bootstrap.py
@Software: vscode
@Description:
        针对收集的国内工人工厂噪声暴露的数据进行二分类的逻辑回归的Bootstrap，获取excess risk的置信区间
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

from utils.data_helper import mark_group_name
from Chinese_logistic_regression import logistic_func_original, log_likelihood_original, log_likelihood_indepphi
from Chinese_logistic_regression_control_data_0 import logistic_func as logistic_func_control_0

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
    max_LAeq: float,
    params_init: list,
    L_control: float,
    phi_range: list = None,
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
    minimize_bounds = kwargs.pop(
        "minimize_bounds",
        ([None, None], [None, None], [1, 4], [4, 6], [6, 9]))

    log_likelihood_value = []
    params_estimated = []
    phi_estimated = []
    work_df = resample_df.copy()
    work_df = pd.get_dummies(work_df, columns=["duration_box_best"])
    work_df["LAeq"] = work_df["LAeq"].apply(
        lambda x: (x - L_control) / (max_LAeq - L_control) if x != 0 else 0)
    work_df["duration_box_best_D-1"] *= work_df["LAeq"]
    work_df["duration_box_best_D-2"] *= work_df["LAeq"]
    work_df["duration_box_best_D-3"] *= work_df["LAeq"]

    y = work_df[y_col_name]
    X = work_df.drop(columns=[y_col_name, "LAeq"])
    if phi_range:
        for phi in phi_range:
            results = minimize(log_likelihood_indepphi,
                               params_init,
                               args=(phi, X.values, y.values),
                               method=minimize_method,
                               options=minimize_options,
                               bounds=minimize_bounds)
            log_likelihood_value.append(results.fun)
            params_estimated.append(results.x)
            phi_estimated.append(phi)
    else:
        results = minimize(log_likelihood_original,
                           params_init,
                           args=(X.values, y.values),
                           method=minimize_method,
                           options=minimize_options,
                           bounds=minimize_bounds)
        log_likelihood_value.append(results.fun)
        params_estimated.append(results.x)

    best_log_likelihood_value = np.min(log_likelihood_value)
    best_params_estimated = params_estimated[np.argmin(log_likelihood_value)]
    if phi_range:
        best_phi_estimated = phi_estimated[np.argmin(log_likelihood_value)]
        best_params_estimated = np.append(best_params_estimated,
                                          best_phi_estimated)
    best_L_control = L_control

    return best_params_estimated, best_L_control, best_log_likelihood_value


def get_excess_risk(base_params_estimated: list,
                    base_L_control: list,
                    max_LAeq: int,
                    control_params_estimated: list,
                    age: int = 30,
                    LAeq: np.array = np.arange(70, 100, 10),
                    duration: np.array = np.array([1, 0, 0])):
    LAeq_duration_matrix = np.tile(duration, (len(LAeq), 1)) * (
        (LAeq - base_L_control) / (max_LAeq - base_L_control))[:, np.newaxis]
    age_matrix = age * np.ones(len(LAeq))

    estimate_X = np.concatenate(
        (age_matrix[:, np.newaxis], LAeq_duration_matrix), axis=1)
    excess_risk = logistic_func_original(x=estimate_X,
                                         params=base_params_estimated)
    control_X = np.array([[age]])
    background_risk = logistic_func_control_0(x=control_X,
                                              params=control_params_estimated)
    excess_risk -= background_risk
    return pd.DataFrame(data=excess_risk, index=LAeq)


def bootstrap_estimates(data: pd.DataFrame,
                        base_params_estimated: list,
                        base_L_control: float,
                        max_LAeq: int,
                        control_params_estimated: list,
                        y_col_name: str,
                        n_iterations: int = 1000,
                        age: int = 30,
                        LAeq: np.array = np.arange(70, 100, 10),
                        duration: np.array = np.array([1, 0, 0]),
                        **kwargs):
    minimize_method = kwargs.pop(
        "minimize_method", "Nelder-Mead"
    )  #"SLSQP", #"Powell", #"L-BFGS-B", #"Nelder-Mead", #"BFGS",
    minimize_options = kwargs.pop("minimize_options", {'maxiter': 10000})
    minimize_bounds = kwargs.pop(
        "minimize_bounds",
        ([None, None], [None, None], [1, 4], [4, 6], [6, 9]))

    estimates = []
    for i in tqdm(range(n_iterations)):
        resample_df = resample(data, replace=True, n_samples=data.shape[0])
        best_params_estimated, best_L_control, _ = userdefine_logistic_regression_Bootstrap_task(
            resample_df=resample_df,
            max_LAeq=max_LAeq,
            params_init=base_params_estimated,
            # phi_range=[base_params_estimated[-1]],
            L_control=base_L_control,
            minimize_method=minimize_method,
            minimize_options=minimize_options,
            minimize_bounds=minimize_bounds,
            y_col_name=y_col_name)
        LAeq_duration_matrix = np.tile(duration, (len(LAeq), 1)) * (
            (LAeq - best_L_control) /
            (max_LAeq - best_L_control))[:, np.newaxis]
        age_matrix = age * np.ones(len(LAeq))

        estimate_X = np.concatenate(
            (age_matrix[:, np.newaxis], LAeq_duration_matrix), axis=1)
        estimate_y = logistic_func_original(x=estimate_X,
                                            params=best_params_estimated)
        control_X = np.array([[age]])
        control_y = logistic_func_control_0(x=control_X,
                                            params=control_params_estimated)
        estimate_y -= control_y
        estimates.append(dict(zip(LAeq, estimate_y)))
    return pd.DataFrame(estimates)


def confidence_limit_plot(plot_dfs: list, key_point_xs: list, age: int,
                          duration: np.array, picture_path: Path,
                          picture_name: str, picture_format: str, annotations={"A": (-0.1, 1.05)}, **kwargs):
    freq_cols = kwargs.pop("freq_cols", ["NIHL1234w_Y", "NIHL346_Y"])
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

    fig, ax = plt.subplots(1, figsize=(6.5, 5), dpi=dpi)
    for plot_df, freq_col, color in zip(plot_dfs, freq_cols,
                                        ['#1f77b4', '#ff7f0e']):
        if freq_col in ["NIHL1234_Y", "NIHL1234w_Y"]:
            label_name = "HL$_{1234}$"
        elif freq_col == "NIHL346_Y":
            label_name = "HL$_{346}$"
        else:
            label_name = "HL"

        ax.plot(plot_df.index,
                plot_df.excess_risk * 100,
                label=f"{label_name}",
                color=color)
        ax.fill_between(plot_df.index,
                        y1=plot_df.lower_bounds * 100,
                        y2=plot_df.upper_bounds * 100,
                        color=color,
                        alpha=0.4)
        x_min, x_max = ax.get_xlim()
        if y_lim:
            y_min, y_max = y_lim
        else:
            y_min, y_max = ax.get_ylim()
        for key_point_x in key_point_xs:
            key_point_y = plot_df.loc[key_point_x]["excess_risk"] * 100
            ax.vlines(x=key_point_x,
                      ymin=y_min,
                      ymax=key_point_y,
                      colors="black",
                      linestyles=":")
            ax.annotate("{:.2f}".format(key_point_y),
                        xy=(key_point_x, key_point_y),
                        xytext=(key_point_x - (x_max - x_min) / 10,
                                key_point_y + (y_max - y_min) / 20),
                        color=color,
                        arrowprops=dict(color=color,
                                        arrowstyle="->",
                                        linestyle="--"))
    ax.set_ylim(y_min, y_max)
    ax.set_ylabel("Excess Risk of Hearing Loss (%)")
    ax.set_xlabel("$L_{Aeq,8h}$ (dBA)")
    ax.set_title(f"Age = {age}, Duration {duration_desp}")

    plt.legend(loc="upper left")
    for label, (x, y) in annotations.items():
        ax.annotate(label,
                    xy=(1, 0),
                    xycoords='axes fraction',
                    xytext=(x, y),
                    textcoords='axes fraction',
                    fontproperties=FontProperties(size=20, weight='bold'))
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
        f"./log/Chinese_logistic_regression_Bootstrap-{datetime.now().strftime('%Y-%m-%d')}.log",
        level="INFO")

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_path",
        type=str,
        default="./cache/Chinese_extract_experiment_df_average_freq_1234w.csv")
    # default="./cache/Chinese_extract_experiment_df_average_freq.csv")
    # default="./cache/Chinese_extract_experiment_classifier_df.csv")
    # parser.add_argument("--task", type=str, default="analysis")
    parser.add_argument("--task", type=str, default="plot")
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

        # data type convert
        duration_cut = [0, 4, 10, np.inf]
        extract_df["duration_box_best"] = extract_df["duration"].apply(
            lambda x: mark_group_name(x, qcut_set=duration_cut, prefix="D-"))

        # freq_col = "NIHL1234w_Y"
        freq_col = "NIHL346_Y"

        # age = 30
        # duration = np.array([0, 1, 0])
        # age = 45
        # duration = np.array([0, 1, 0])
        # age = 45
        # duration = np.array([0, 0, 1])
        age = 65
        duration = np.array([0, 0, 1])

        fit_df = extract_df.query(
            "duration_box_best in ('D-1', 'D-2', 'D-3') and LAeq >= 70")[[
                "age", "LAeq", "duration_box_best", freq_col
            ]]
        base_params_estimated, base_L_control, max_LAeq, best_log_likelihood_value = pickle.load(
            open(
                models_path / Path(
                    f"{freq_col}-Chinese_experiment_group_udlr_model_average_freq.pkl"
                ), "rb"))
        control_params_estimated, control_log_likelihood_value = pickle.load(
            open(
                models_path /
                Path(f"{freq_col[2:]}-NOISH_control_group_udlr_model_0.pkl"),
                "rb"))
        # base_params_estimated = [-4.99, 0.08, 8.0791, 8.9607, 9.9687, 3.0]
        # base_L_control = 60

        excess_risk_value = get_excess_risk(
            base_params_estimated=base_params_estimated,
            base_L_control=base_L_control,
            max_LAeq=max_LAeq,
            control_params_estimated=control_params_estimated,
            age=age,
            LAeq=np.arange(60, 101),
            duration=duration)
        estimates = bootstrap_estimates(
            data=fit_df,
            base_params_estimated=base_params_estimated,
            base_L_control=base_L_control,
            max_LAeq=max_LAeq,
            control_params_estimated=control_params_estimated,
            y_col_name=freq_col,
            n_iterations=1000,
            age=age,
            LAeq=np.arange(60, 101),
            duration=duration,
            minimize_bounds=([None, -4.7], [None, 0.09], [None, None],
                             [None, None], [None, None], [1, 3]))
        # minimize_bounds=([-5.50, -4.99], [0.07, 0.08], [None, None],
        #                  [None, None], [None, None], [1, 3]))

        confidence_res = pd.DataFrame()
        confidence_res["excess_risk"] = excess_risk_value
        confidence_res["lower_bounds"] = estimates.apply(
            lambda x: np.percentile(x, 5))
        confidence_res["upper_bounds"] = estimates.apply(
            lambda x: np.percentile(x, 95))

        confidence_res.to_csv(
            output_path /
            f"{freq_col[:-2]}-Chinese_conf_limit_{age}-{str(duration[-1])}.csv",
            header=True,
            index=True)

    if task == "plot":
        freq_col_1 = "NIHL1234_Y"
        freq_col_2 = "NIHL346_Y"

        # age = 30
        # duration = np.array([0, 1, 0])
        # age = 45
        # duration = np.array([0, 1, 0])
        age = 45
        duration = np.array([0, 0, 1])
        # age = 65
        # duration = np.array([0, 0, 1])

        confidence_res_1 = pd.read_csv(
            output_path /
            f"{freq_col_1[:-2]}-Chinese_conf_limit_{age}-{str(duration[-1])}.csv",
            header=0,
            index_col=0)
        confidence_res_2 = pd.read_csv(
            output_path /
            f"{freq_col_2[:-2]}-Chinese_conf_limit_{age}-{str(duration[-1])}.csv",
            header=0,
            index_col=0)

        confidence_limit_plot(
            plot_dfs=[confidence_res_1, confidence_res_2],
            key_point_xs=[80, 85, 90, 95, 100],
            age=age,
            duration=duration,
            picture_path=pictures_path,
            picture_name="Fig3C",
            # f"{freq_col[:-2]}-Chinese_conf_limit_{age}-{str(duration[-1])}",
            picture_format="tiff",
            dpi=330,
            freq_cols=[freq_col_1, freq_col_2],
            y_lim=[-2, 60],
            annotations={"C": (-0.1, 1.05)})
    print(1)
