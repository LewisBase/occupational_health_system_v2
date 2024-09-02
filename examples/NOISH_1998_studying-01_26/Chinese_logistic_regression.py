# -*- coding: utf-8 -*-
"""
@DATE: 2024-01-31 09:51:22
@Author: Liu Hengjiang
@File: examples\\NOISH_1998_studying-01_26\Chinese_logistic_regression.py
@Software: vscode
@Description:
        针对收集的国内工人工厂噪声暴露的数据进行二分类的逻辑回归尝试
"""

import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from pathlib import Path
from functional import seq
from loguru import logger
from joblib import Parallel, delayed
from scipy.optimize import minimize

from staff_info import StaffInfo
from diagnose_info.auditory_diagnose import AuditoryDiagnose
from utils.data_helper import mark_group_name, filter_data
from Chinese_logistic_regression_control_data_0 import logistic_func as logistic_func_control_0
from Chinese_logistic_regression_control_data_1 import logistic_func as logistic_func_control_1

from matplotlib.font_manager import FontProperties
from matplotlib import rcParams
from matplotlib.lines import Line2D

config = {
    "font.family": "serif",
    "font.size": 12,
    "mathtext.fontset": "stix",  # matplotlib渲染数学字体时使用的字体，和Times New Roman差别不大
    "font.serif": ["STZhongsong"],  # 华文中宋
    "axes.unicode_minus": False  # 处理负号，即-号
}
rcParams.update(config)


def _extract_data_for_task(data, **additional_set):
    NIPTS_diagnose_strategy = additional_set.pop("NIPTS_diagnose_strategy")

    res = {}
    res["staff_id"] = data.staff_id
    # worker information
    res["age"] = data.staff_basic_info.age
    res["sex"] = data.staff_basic_info.sex
    res["duration"] = data.staff_basic_info.duration

    # worker health infomation
    res["NIHL1234"] = data.staff_health_info.auditory_detection.get(
        "PTA").mean(mean_key=[1000, 2000, 3000, 4000])
    res["NIHL346"] = data.staff_health_info.auditory_detection.get("PTA").mean(
        mean_key=[3000, 4000, 6000])
    res["NIPTS1234"] = AuditoryDiagnose.NIPTS(
        detection_result=data.staff_health_info.auditory_detection["PTA"],
        sex=data.staff_basic_info.sex,
        age=data.staff_basic_info.age,
        mean_key=[1000, 2000, 3000, 4000],
        NIPTS_diagnose_strategy=NIPTS_diagnose_strategy)
    res["NIPTS346"] = AuditoryDiagnose.NIPTS(
        detection_result=data.staff_health_info.auditory_detection["PTA"],
        sex=data.staff_basic_info.sex,
        age=data.staff_basic_info.age,
        mean_key=[3000, 4000, 6000],
        NIPTS_diagnose_strategy=NIPTS_diagnose_strategy)

    res["NIHL1234_Y"] = 0 if res["NIHL1234"] <= 25 else 1
    res["NIHL346_Y"] = 0 if res["NIHL346"] <= 25 else 1
    res["NIPTS1234_Y"] = 0 if res["NIPTS1234"] <= 0 else 1
    res["NIPTS346_Y"] = 0 if res["NIPTS346"] <= 0 else 1

    # noise information
    res["LAeq"] = data.staff_occupational_hazard_info.noise_hazard_info.LAeq
    res["kurtosis_arimean"] = data.staff_occupational_hazard_info.noise_hazard_info.kurtosis_arimean
    res["kurtosis_geomean"] = data.staff_occupational_hazard_info.noise_hazard_info.kurtosis_geomean

    return res


def extract_data_for_task(df, n_jobs=-1, **additional_set):
    res = Parallel(n_jobs=n_jobs, backend="multiprocessing")(
        delayed(_extract_data_for_task)(data=data, **additional_set)
        for data in df)
    res = pd.DataFrame(res)
    return res


def logistic_func_original(params, x):
    alpha, beta1, beta21, beta22, beta23, phi = params
    F = alpha + beta1 * x[:, 0] + beta21 * np.power(
        x[:, 1], phi) + beta22 * np.power(x[:, 2], phi) + beta23 * np.power(
            x[:, 3], phi)
    return np.exp(F) / (1 + np.exp(F))


def logistic_func_indepphi(params, phi, x):
    alpha, beta1, beta21, beta22, beta23 = params
    F = alpha + beta1 * x[:, 0] + beta21 * np.power(
        x[:, 1], phi) + beta22 * np.power(x[:, 2], phi) + beta23 * np.power(
            x[:, 3], phi)
    return np.exp(F) / (1 + np.exp(F))


def log_likelihood_original(params, x, y):
    p = logistic_func_original(params, x)
    log_likelihood = -1 * np.sum(y * np.log(p) + (1 - y) * np.log(1 - p))
    return log_likelihood


def log_likelihood_indepphi(params, phi, x, y):
    p = logistic_func_indepphi(params, phi, x)
    log_likelihood = -1 * np.sum(y * np.log(p) + (1 - y) * np.log(1 - p))
    return log_likelihood


def userdefine_logistic_regression_task(
        fit_df: pd.DataFrame,
        max_LAeq: float,
        models_path: Path,
        model_name: str,
        y_col_name: str = "NIHL1234_Y",
        params_init: list = [
            -4.18049946, 0.07176393, 1.32653869, 2.13749184, 8.65684751, 3
        ],
        L_control_range: np.array = np.arange(70, 90),
        phi_range=None,
        **kwargs,
):
    """_summary_
    自定义逻辑回归函数

    Args:
        fit_df (pd.DataFrame): _description_
        models_path (Path): _description_
        model_name (str): _description_
        y_col_name (str, optional): _description_. Defaults to "NIHL1234_Y".
        params_init (list, optional): _description_. Defaults to [ -4.18049946, 0.07176393, 1.32653869, 2.13749184, 8.65684751, 3 ].
        L_control_range (np.array, optional): _description_. Defaults to np.arange(70, 90).
        phi_range
    """

    minimize_method = kwargs.pop(
        "minimize_method", "Nelder-Mead"
    )  #"SLSQP", #"Powell", #"L-BFGS-B", #"Nelder-Mead", #"BFGS",
    minimize_options = kwargs.pop("minimize_options", {'maxiter': 10000})
    minimize_bounds = kwargs.pop("minimize_bounds", None)

    logger.info(f"minimize method: {minimize_method}")
    logger.info(f"minimize bounds: {minimize_bounds}")

    log_likelihood_value = []
    params_estimated = []
    phi_estimated = []
    for L_control in L_control_range:
        # logger.info(f"Fit result for L_control = {L_control}")
        work_df = fit_df.copy()
        work_df = pd.get_dummies(work_df, columns=["duration_box_best"])
        work_df["LAeq"] = work_df["LAeq"].apply(lambda x: (x - L_control) / (
            max_LAeq - L_control) if x != 0 else 0)
        # work_df["LAeq"] -= L_control
        # work_df["LAeq"] /= (max_LAeq - L_control)
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
                # logger.info(f"Fixed phi: {phi}")
                # logger.info(f"Fit status: {results.success}")
                # logger.info(f"Log likehood: {round(results.fun,2)}")
                # logger.info(f"Iterations: {results.nit}")
                # logger.info(f"Fit parameters: {results.x}")
        else:
            results = minimize(log_likelihood_original,
                               params_init,
                               args=(X.values, y.values),
                               method=minimize_method,
                               options=minimize_options,
                               bounds=minimize_bounds)
            log_likelihood_value.append(results.fun)
            params_estimated.append(results.x)
            # logger.info(f"Fit status: {results.success}")
            # logger.info(f"Log likehood: {round(results.fun,2)}")
            # logger.info(f"Iterations: {results.nit}")
            # logger.info(f"Fit parameters: {results.x}")
            

    best_log_likelihood_value = np.min(log_likelihood_value)
    best_params_estimated = params_estimated[np.argmin(log_likelihood_value)]
    if phi_range:
        best_phi_estimated = phi_estimated[np.argmin(log_likelihood_value)]
        best_params_estimated = np.append(best_params_estimated,
                                          best_phi_estimated)
        best_L_control = L_control_range[np.argmin(log_likelihood_value) // len(phi_range)]
    else:
        best_L_control = L_control_range[np.argmin(log_likelihood_value)]
    logger.info(
        f"Final result: {seq(best_params_estimated).map(lambda x: round(x,4))} + {best_L_control}. \n Log likelihood: {round(best_log_likelihood_value,2)}"
    )

    pickle.dump([
        best_params_estimated, best_L_control, max_LAeq,
        best_log_likelihood_value
    ], open(models_path / Path(y_col_name + "-" + model_name), "wb"))
    return best_params_estimated, best_L_control, max_LAeq, best_log_likelihood_value


def userdefine_logistic_regression_plot(best_params_estimated,
                                        best_L_control,
                                        max_LAeq,
                                        picture_path: Path,
                                        picture_name: str,
                                        picture_format: str,
                                        age: int = 30,
                                        LAeq: np.array = np.arange(70, 100),
                                        duration: np.array = np.array(
                                            [1, 0, 0]),
                                        point_type: str = "2nd",
                                        **kwargs):
    """_summary_
    概率曲线绘制函数

    Args:
        best_params_estimated (_type_): _description_
        best_L_control (_type_): _description_
        max_LAeq (_type_): _description_
        age (int, optional): _description_. Defaults to 30.
        LAeq (np.array, optional): _description_. Defaults to np.arange(70, 100).
        duration (np.array, optional): _description_. Defaults to np.array([1, 0, 0]).
        point_type (str, optional): _description_. Defaults to "2nd".

    Raises:
        ValueError: _description_
    """

    plot_der = kwargs.pop("plot_der", True)
    control_params_estimated = kwargs.pop("control_params_estimated", None)
    control_y = kwargs.pop("y_control", 0)
    dpi = kwargs.pop("dpi", 330)
    is_show = kwargs.pop("is_show", False)
    y_lim = kwargs.pop("y_lim", None)
    freq_col = kwargs.pop("freq_col", "NIHL1234_Y")

    if duration[0] == 1:
        duration_desp = "= 1~4"
    elif duration[1] == 1:
        duration_desp = "= 5~10"
    elif duration[2] == 1:
        duration_desp = "> 10"
    else:
        raise ValueError
    if freq_col == "NIHL1234_Y":
        # label_name = "$HL_{1234}$"
        label_name = "$\\text{HL}_{1234}$"
    elif freq_col == "NIHL346_Y":
        # label_name = "$HL_{346}$"
        label_name = "$\\text{HL}_{346}$"
    else:
        # label_name = "$HL$"
        label_name = "$\\text{HL}$"
        
    logger.info(f"Params: {best_params_estimated}")
    logger.info(f"L_control: {best_L_control}")
    logger.info(f"max LAeq: {max_LAeq}")

    LAeq_duration_matrix = np.tile(duration, (len(LAeq), 1)) * (
        (LAeq - best_L_control) / (max_LAeq - best_L_control))[:, np.newaxis]
    age_matrix = age * np.ones(len(LAeq))

    plot_X = np.concatenate((age_matrix[:, np.newaxis], LAeq_duration_matrix),
                            axis=1)
    pred_y = logistic_func_original(x=plot_X, params=best_params_estimated)
    f_prime = np.gradient(pred_y, LAeq)
    f_prime_double = np.gradient(f_prime, LAeq)
    # logger.info(f"f prime: {f_prime}")
    # logger.info(f"f prime double: {f_prime_double}")
    

    if point_type == "1st":
        point_x = LAeq[np.nanargmax(f_prime)]
    elif point_type == "2nd":
        point_x = LAeq[np.nanargmax(f_prime_double)]
    elif point_type == "2nd_critical":
        f_prime_double_filled_nan = np.nan_to_num(f_prime_double, nan=+0.0)
        logger.info(f"f prime double filled nan: {f_prime_double_filled_nan}")
        diff_arr = np.diff(np.sign(f_prime_double_filled_nan))
        point_x = LAeq[np.where(diff_arr != 0)[0] + 1][0]

    age_array = np.array([age])
    point_x_duration_array = (point_x - best_L_control) / (
        max_LAeq - best_L_control) * duration
    point_X = np.concatenate((age_array, point_x_duration_array),
                             axis=0)[np.newaxis, :]
    point_y = logistic_func_original(x=point_X, params=best_params_estimated)

    if control_params_estimated is not None:
        if len(control_params_estimated) == 2:
            control_X = np.array([[age]])
            control_y = logistic_func_control_0(
                x=control_X, params=control_params_estimated)
        else:
            control_X = np.array([np.concatenate([[age], duration])])
            control_y = logistic_func_control_1(
                x=control_X, params=control_params_estimated)
        logger.info(f"control base probability: {control_y}")
        logger.info(f"excess risk values: {(pred_y - control_y)[np.where((LAeq==80)|(LAeq==85)|(LAeq==90)|(LAeq==95)|(LAeq==100))]}")

    fig, ax = plt.subplots(1, figsize=(6.5, 5))
    ax.plot(LAeq, (pred_y - control_y)*100)
    if y_lim:
        y1_min, y1_max = y_lim
    else:
        y1_min, y1_max = ax.get_ylim()
    # ax.set_ylim(-0.05, y1_max)
    ax.annotate(f"key point: {point_x} dBA",
                xy=(point_x, (point_y - control_y)*100),
                xytext=(point_x - (max(LAeq) - min(LAeq)) / 5,
                        (point_y - control_y)*100 + (y1_max - y1_min) / 10),
                color="red",
                arrowprops=dict(color="red", arrowstyle="->"))
    ax.set_ylim(y1_min, y1_max)
    ax.set_title(f"Age = {age}, Duration {duration_desp}")
    ax.set_ylabel("Excess Risk of Hearing Loss (%)")
    ax.set_xlabel("$L_{Aeq,8h}$ (dBA)")
    if plot_der:
        ax2 = ax.twinx()
        ax2.plot(LAeq, f_prime, "c--", label="1st derivative", alpha=0.4)
        ax2.plot(LAeq,
                 f_prime_double,
                 "g--",
                 label="2nd derivative",
                 alpha=0.4)
        ax2.vlines(x=point_x,
                   ymin=0,
                   ymax=max(f_prime_double),
                   color="black",
                   linestyles=":")
        ax2.hlines(y=0,
                   xmin=min(LAeq),
                   xmax=max(LAeq),
                   colors="black",
                   linestyles=":")
        ax2.tick_params(axis="y", colors="c")
        line1 = Line2D([], [], color="c", linestyle="--", alpha=0.4, label=f"{label_name} 1st derivative")
        line2 = Line2D([], [], color="g", linestyle="--", alpha=0.4, label=f"{label_name} 2nd derivative")
    
    # 创建图例对象
    line0 = Line2D([], [], color="#1f77b4", label=f"{label_name}")
    if line1 and line2:
        legend = plt.legend(handles=[line0, line1, line2], loc="upper left")
    else:
        legend = plt.legend(handles=[line0], loc="upper left")
    # 添加图例到图形
    plt.gca().add_artist(legend)

    plt.tight_layout()
    picture_path = Path(picture_path) / f"{picture_name}.{picture_format}"
    plt.savefig(picture_path, format=picture_format, dpi=dpi)
    if is_show:
        plt.show()
    plt.close(fig=fig)
    return dict(zip(LAeq, pred_y))


def logistic_vector_func(params, age_grid, LAeq_grid, duration):
    alpha, beta1, beta21, beta22, beta23, phi = params
    if duration[0] == 1:
        beta2_use = beta21
    if duration[1] == 1:
        beta2_use = beta22
    if duration[2] == 1:
        beta2_use = beta23
    F = alpha + beta1 * age_grid + beta2_use * np.power(LAeq_grid, phi)
    return F


def userdefine_logistic_vector_plot(best_params_estimated, best_L_control,
                                    max_LAeq, picture_name, pictures_path):
    """根据回归结果绘制三维图

    Args:
        best_params_estimated (_type_): _description_
        best_L_control (_type_): _description_
        max_LAeq (_type_): _description_
        picture_name (_type_): _description_
        pictures_path (_type_): _description_
    """
    picture_path = Path(pictures_path) / Path(picture_name)
    ages = np.linspace(15, 61, 100)
    LAeqs = np.linspace(70, 100, 100)
    LAeqs = (LAeqs - best_L_control) / (max_LAeq - best_L_control)
    age_grid, LAeq_grid = np.meshgrid(ages, LAeqs)
    fig = go.Figure()
    durations = {
        "Duration = 1~4": [1, 0, 0],
        "Duration = 5~10": [0, 1, 0],
        "Duration > 10": [0, 0, 1],
    }
    for duration_name, duration in durations.items():
        vector_z = logistic_vector_func(params=best_params_estimated,
                                        age_grid=age_grid,
                                        LAeq_grid=LAeq_grid,
                                        duration=duration)
        fig.add_trace(
            go.Surface(x=ages,
                       y=LAeqs,
                       z=vector_z,
                       text=duration_name,
                       hoverinfo="text"))
        fig.update_layout(scene=dict(xaxis=dict(title="Age (year)"),
                                     yaxis=dict(title="$L_{Aeq}$ (dBA)"),
                                     zaxis=dict(
                                         title="explanatory variables value")))
        fig.write_html(picture_path)


def userdefine_logistic_vector_compare_plot(best_params_estimated_1,
                                            best_L_control_1,
                                            max_LAeq_1, 
                                            best_params_estimated_2,
                                            best_L_control_2,
                                            max_LAeq_2,
                                            duration,
                                            picture_name, 
                                            pictures_path):
    """根据回归结果绘制三维对比图

    Args:
        best_params_estimated_1 (_type_): _description_
        best_L_control_1 (_type_): _description_
        max_LAeq_1 (_type_): _description_
        best_params_estimated_2 (_type_): _description_
        best_L_control_2 (_type_): _description_
        max_LAeq_2 (_type_): _description_
        duration (_type_): _description_
        picture_name (_type_): _description_
        pictures_path (_type_): _description_
    """
    picture_path = Path(pictures_path) / Path(picture_name)
    ages = np.linspace(15, 61, 100)
    LAeqs = np.linspace(70, 100, 100)
    LAeqs_1 = (LAeqs - best_L_control_1) / (max_LAeq_1 - best_L_control_1)
    LAeqs_2 = (LAeqs - best_L_control_2) / (max_LAeq_2 - best_L_control_2)
    age_grid, LAeq_1_grid = np.meshgrid(ages, LAeqs_1)
    age_grid, LAeq_2_grid = np.meshgrid(ages, LAeqs_2)

    vector_z_1 = logistic_vector_func(params=best_params_estimated_1,
                                      age_grid=age_grid,
                                      LAeq_grid=LAeq_1_grid,
                                      duration=duration)
    vector_z_2 = logistic_vector_func(params=best_params_estimated_2,
                                      age_grid=age_grid,
                                      LAeq_grid=LAeq_2_grid,
                                      duration=duration)

    fig = go.Figure()
    fig.add_trace(
        go.Surface(x=ages,
                   y=LAeqs,
                   z=vector_z_1,
                   text="NIHL1234",
                   hoverinfo="text"))
    fig.add_trace(
        go.Surface(x=ages,
                   y=LAeqs,
                   z=vector_z_2,
                   text="NIHL346",
                   hoverinfo="text"))
    fig.update_layout(scene=dict(xaxis=dict(title="Age (year)"),
                                 yaxis=dict(title=r"$L_{Aeq}$ (dBA)"),
                                 zaxis=dict(
                                     title="explanatory variables value")))
    fig.write_html(picture_path)


if __name__ == "__main__":
    from datetime import datetime
    logger.add(
        f"./log/Chinese_logistic_regression-{datetime.now().strftime('%Y-%m-%d')}.log",
        level="INFO")

    import argparse
    parser = argparse.ArgumentParser()
    # parser.add_argument("--input_path",
    #                     type=str,
    #                     default="./cache/extract_Chinese_data_average_freq.pkl")
    #                     # default="./cache/extract_Chinese_data.pkl")
    # parser.add_argument("--task", type=str, default="extract")
    parser.add_argument(
        "--input_path",
        type=str,
        default="./cache/Chinese_extract_experiment_df_average_freq.csv")
        # default="./cache/Chinese_extract_experiment_classifier_df.csv")
    parser.add_argument("--task", type=str, default="analysis")
    # parser.add_argument("--task", type=str, default="plot")
    parser.add_argument("--output_path", type=str, default="./cache")
    parser.add_argument("--models_path", type=str, default="./models")
    parser.add_argument("--pictures_path", type=str, default="./pictures")
    parser.add_argument("--additional_set",
                        type=dict,
                        default={
                            "NIPTS_diagnose_strategy": "better"
                        })
    parser.add_argument(
        "--annotated_bad_case",
        type=list,
        default=[
            "沃尔夫链条-60",
            "杭州重汽发动机有限公司-10",
            "浙江红旗机械有限公司-6",
            "Wanhao furniture factory-41",
            "Songxia electrical appliance factory-40",
            "Songxia electrical appliance factory-18",
            "Songxia electrical appliance factory-15",
            "Mamibao baby carriage manufactory-77",
            "Liyuan hydroelectric-51",
            "Liyuan hydroelectric-135",
            "Liyuan hydroelectric-112",
            "Liyuan hydroelectric-103",
            "Huahui Machinery-11",
            "Hebang brake pad manufactory-95",
            "Hebang brake pad manufactory-94",
            "Gujia furniture factory-9",
            "Gujia furniture factory-85",
            "Gujia furniture factory-54",
            "Gujia furniture factory-5",
            "Gujia furniture factory-39",
            "Gujia furniture factory-35",
            "Gengde electronic equipment factory-57",
            "Gengde electronic equipment factory-47",
            "Changhua Auto Parts Manufactory-6",
            "Changhua Auto Parts Manufactory-127",
            "Botai furniture manufactory-17",
            "Banglian spandex-123",
            "Changhua Auto Parts Manufactory-40",
            "Banglian spandex-12",
            "Changhua Auto Parts Manufactory-270",
            "Changhua Auto Parts Manufactory-48",
            "Gujia furniture factory-35",
            "Hebang brake pad manufactory-165",
            "Hebang brake pad manufactory-20",
            "Hengfeng paper mill-31",
            "Liyuan hydroelectric-135",
            "Liyuan hydroelectric-30",
            "NSK Precision Machinery Co., Ltd-109",
            "NSK Precision Machinery Co., Ltd-345",
            "Songxia electrical appliance factory-15",
            "Waigaoqiao Shipyard-170",
            "Waigaoqiao Shipyard-94",
            "春风动力-119",
            "浙江红旗机械有限公司-20",
            "浙江红旗机械有限公司-5",
            "Banglian spandex-123",
            "Botai furniture manufactory-66",
            "Changhua Auto Parts Manufactory-120",
            "Changhua Auto Parts Manufactory-141",
            "Changhua Auto Parts Manufactory-355",
            "Changhua Auto Parts Manufactory-40",
            "Gujia furniture factory-39",
            "Gujia furniture factory-5",
            "Gujia furniture factory-85",
            "Hengfeng paper mill-27",
            "Hengjiu Machinery-15",
            "Liyuan hydroelectric-120",
            "Liyuan hydroelectric-14",
            "NSK Precision Machinery Co., Ltd-288",
            "NSK Precision Machinery Co., Ltd-34",
            "Yufeng paper mill-26",
            "春风动力-98",
            "春江-1",
            "东华链条厂-60",
            "东华链条厂-77",
            "东华链条厂-79",
            "双子机械-9",
            "沃尔夫链条-59",
            "中国重汽杭州动力-83",
            "Wanhao furniture factory-24",
            "永创智能-46",
            "Wanhao furniture factory-34",
            "永创智能-45",
            "总装配厂-117",
            "总装配厂-467",
            "东风汽车有限公司商用车车身厂-259",
            "东风汽车紧固件有限公司-405",
            "东风汽车车轮有限公司-16",
            "Huahui Machinery-10",
            "Gujia furniture factory-3",
            # 原来用来修改的一条记录，这里直接去掉
            "东风汽车有限公司商用车车架厂-197",
        ])
    parser.add_argument("--n_jobs", type=int, default=-1)
    args = parser.parse_args()

    logger.info("Input Parameters informations:")
    message = "\n".join([f"{k:<20}: {v}" for k, v in vars(args).items()])
    logger.info(message)

    input_path = Path(args.input_path)
    output_path = Path(args.output_path)
    models_path = Path(args.models_path)
    pictures_path = Path(args.pictures_path)
    additional_set = args.additional_set
    annotated_bad_case = args.annotated_bad_case
    task = args.task
    n_jobs = args.n_jobs
    for out_path in (output_path, models_path, pictures_path):
        if not out_path.exists():
            out_path.mkdir(parents=True)

    if task == "extract":
        original_data = pickle.load(open(input_path, "rb"))
        original_data = seq(original_data).flatten().list()
        extract_df = extract_data_for_task(df=original_data,
                                           n_jobs=n_jobs,
                                           **additional_set)
        filter_df = filter_data(
            df_total=extract_df,
            drop_col=None,
            dropna_set=["NIHL1234", "LAeq", "kurtosis_geomean"],
            str_filter_dict={"staff_id": annotated_bad_case},
            num_filter_dict={
                "age": {
                    "up_limit": 65,
                    # "up_limit": 60,
                    "down_limit": 15
                },
                # "LAeq": {
                #     "up_limit": 200,
                #     "down_limit": 80
                # },
            },
            eval_set=None)

        filter_df.index = filter_df.staff_id
        filter_df.drop("staff_id", axis=1, inplace=True)
        filter_df.to_csv(output_path /
                         "Chinese_extract_experiment_df_average_freq.csv",
                        #  "Chinese_extract_experiment_classifier_df.csv",
                         header=True,
                         index=True)
    if task == "analysis":
        extract_df = pd.read_csv(input_path, header=0, index_col="staff_id")

        # data type convert
        duration_cut = [0, 4, 10, np.inf]
        kurtosis_arimean_cut = [3, 10, 50, np.inf]
        kurtosis_geomean_cut = [3, 25, 60, 160]
        extract_df["duration_box_best"] = extract_df["duration"].apply(
            lambda x: mark_group_name(x, qcut_set=duration_cut, prefix="D-"))
        extract_df["kurtosis_arimean_box"] = extract_df[
            "kurtosis_arimean"].apply(lambda x: mark_group_name(
                x, qcut_set=kurtosis_arimean_cut, prefix="KA-"))
        extract_df["kurtosis_geomean_box"] = extract_df[
            "kurtosis_geomean"].apply(lambda x: mark_group_name(
                x, qcut_set=kurtosis_geomean_cut, prefix="KG-"))

        # 使用全部数据
        # ## NIHL1234_Y
        # max_LAeq = extract_df["LAeq"].max()
        # fit_df = extract_df.query(
        #     "duration_box_best in ('D-1', 'D-2', 'D-3') and LAeq >= 70")[[
        #         "age", "LAeq", "duration_box_best", "NIHL1234_Y"
        #     ]]
        # userdefine_logistic_regression_task(
        #     fit_df=fit_df,
        #     max_LAeq=max_LAeq,
        #     models_path=models_path,
        #     model_name="Chinese_experiment_group_udlr_model_average_freq.pkl",
        #     y_col_name="NIHL1234_Y",
        #     params_init=[-5.36, 0.08, 2.66, 3.98, 6.42, 3],
        #     L_control_range=np.arange(60, 79),
        #     # phi_range=[1,2,3],
        #     minimize_bounds = ([-5.50,-4.99],[0.07,0.08],[None, None],[None,None],[None, None],[1, 3]))

        ### KG-groups
        # for KG_group in ["KG-1", "KG-2", "KG-3"]: #, "KG-4"]:
        #     max_LAeq = extract_df["LAeq"].max()
        #     fit_df = extract_df.query(
        #         f"duration_box_best in ('D-1', 'D-2', 'D-3') and LAeq >= 70 and kurtosis_geomean_box == @KG_group")[[
        #             "age", "LAeq", "duration_box_best", "NIHL1234_Y"
        #         ]]
        #     userdefine_logistic_regression_task(
        #         fit_df=fit_df,
        #         max_LAeq=max_LAeq,
        #         models_path=models_path,
        #         model_name=f"{KG_group}-Chinese_experiment_group_udlr_model_average_freq.pkl",
        #         y_col_name="NIHL1234_Y",
        #         params_init=[-5.36, 0.08, 2.66, 3.98, 6.42, 3],
        #         L_control_range=np.arange(60, 79),
        #         # phi_range=[1,2,3],
        #         minimize_bounds = ([-5.50,-4.99],[0.07,0.08],[None, None],[None,None],[None, None],[1, 3]))


        ## NIHL346_Y
        # max_LAeq = extract_df["LAeq"].max()
        # fit_df = extract_df.query(
        #     "duration_box_best in ('D-1', 'D-2', 'D-3') and LAeq >= 70")[[
        #         "age", "LAeq", "duration_box_best", "NIHL346_Y"
        #     ]]
        # userdefine_logistic_regression_task(
        #     fit_df=fit_df,
        #     max_LAeq=max_LAeq,
        #     models_path=models_path,
        #     model_name="Chinese_experiment_group_udlr_model_average_freq.pkl",
        #     y_col_name="NIHL346_Y",
        #     params_init=[-5.36, 0.08, 7.66, 8.98, 9.42, 3],
        #     L_control_range=np.arange(60, 79),
        #     # phi_range=[1,2,3,4,5],
        #     minimize_bounds = ([None, -4.7],[None,0.09],[None, None],[None,None],[None, None],[1, 3]))

        ### KG-groups
        for KG_group in ["KG-1", "KG-2", "KG-3"]:
            max_LAeq = extract_df["LAeq"].max()
            fit_df = extract_df.query(
                f"duration_box_best in ('D-1', 'D-2', 'D-3') and LAeq >= 70 and kurtosis_geomean_box == @KG_group")[[
                    "age", "LAeq", "duration_box_best", "NIHL346_Y"
                ]]
            userdefine_logistic_regression_task(
                fit_df=fit_df,
                max_LAeq=max_LAeq,
                models_path=models_path,
                model_name=f"{KG_group}-Chinese_experiment_group_udlr_model_average_freq.pkl",
                y_col_name="NIHL346_Y",
                params_init=[-5.36, 0.08, 2.66, 3.98, 6.42, 3],
                L_control_range=np.arange(60, 79),
                # phi_range=[1,2,3],
                minimize_bounds = ([None, -4.7],[None,0.09],[None, None],[None,None],[None, None],[1, 3]))

################################################################################################################################
    if task == "plot":
        freq_col = "NIHL1234_Y"
        # freq_col = "NIHL346_Y"

        # KG_group = True
        KG_group = False

        # age = 30
        # duration = np.array([0, 1, 0])
        age = 45 
        duration = np.array([0, 1, 0])
        # age = 45
        # duration = np.array([0, 0, 1])
        # age = 65 
        # duration = np.array([0, 0, 1])

        control_params_estimated, control_log_likelihood_value = pickle.load(
                    open(
                        models_path /
                        Path(f"{freq_col[2:]}-NOISH_control_group_udlr_model_0.pkl"),
                        "rb"))
        
        if KG_group:
            for KG_group in ["KG-1", "KG-2", "KG-3"]:
                best_params_estimated, best_L_control, max_LAeq, best_log_likelihood_value = pickle.load(
                    open(
                        models_path /
                        Path(f"{freq_col}-{KG_group}-Chinese_experiment_group_udlr_model_average_freq.pkl"),
                        "rb"))
                num_res = userdefine_logistic_regression_plot(
                    best_params_estimated=best_params_estimated,
                    best_L_control=best_L_control,
                    max_LAeq=max_LAeq,
                    picture_path=pictures_path,
                    picture_name=f"{freq_col[:-2]}-{KG_group}-Chinese_experiment_excess_risk_{age}-{str(duration[-1])}",
                    picture_format="png",
                    LAeq=np.arange(60, 101),
                    age=age,
                    duration=duration,
                    point_type="2nd",
                    control_params_estimated=control_params_estimated,
                    y_lim=[-2, 60],
                    freq_col=freq_col)
        else:
            best_params_estimated, best_L_control, max_LAeq, best_log_likelihood_value = pickle.load(
                open(
                    models_path /
                    Path(f"{freq_col}-Chinese_experiment_group_udlr_model_average_freq.pkl"),
                    "rb"))
            # best_params_estimated = [-4.7, 0.09, 10.0233, 11.9683, 11.1516, 3.0]
            num_res = userdefine_logistic_regression_plot(
                best_params_estimated=best_params_estimated,
                best_L_control=best_L_control,
                max_LAeq=max_LAeq,
                picture_path=pictures_path,
                picture_name=f"{freq_col[:-2]}-Chinese_experiment_excess_risk_{age}-{str(duration[-1])}",
                picture_format="png",
                LAeq=np.arange(60, 101),
                age=age,
                duration=duration,
                point_type="2nd",
                control_params_estimated=control_params_estimated,
                y_lim=[-2, 60],
                freq_col=freq_col)

    print(1)
