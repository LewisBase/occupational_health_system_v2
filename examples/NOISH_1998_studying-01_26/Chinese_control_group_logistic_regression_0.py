# -*- coding: utf-8 -*-
"""
@DATE: 2024-04-03 15:29:15
@Author: Liu Hengjiang
@File: examples\\NOISH_1998_studying-01_26\Chinese_control_group_logistic_regression_0.py
@Software: vscode
@Description:
        针对收集的部分70dB以下国内工人工厂噪声暴露及对照组的数据进行二分类的逻辑回归尝试
        不考虑duration的，仅以age进行回归
"""

import re
import ast
import pickle
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
from pathlib import Path
from functional import seq
from loguru import logger
from joblib import Parallel, delayed
from sklearn.preprocessing import MinMaxScaler
from scipy.optimize import minimize

from staff_info import StaffInfo
from diagnose_info.auditory_diagnose import AuditoryDiagnose
from utils.data_helper import mark_group_name, filter_data

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


def _extract_data_for_task(data, **additional_set):
    better_ear_strategy = additional_set.pop("better_ear_strategy")
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
    try:
        res["LAeq"] = data.staff_occupational_hazard_info.noise_hazard_info.LAeq
    except AttributeError:
        res["LAeq"] = 0
    # res["kurtosis_arimean"] = data.staff_occupational_hazard_info.noise_hazard_info.kurtosis_arimean
    # res["kurtosis_geomean"] = data.staff_occupational_hazard_info.noise_hazard_info.kurtosis_geomean

    return res


def extract_data_for_task(df, n_jobs=-1, **additional_set):
    res = Parallel(n_jobs=n_jobs, backend="multiprocessing")(
        delayed(_extract_data_for_task)(data=data, **additional_set)
        for data in df)
    res = pd.DataFrame(res)
    return res


def logistic_func(params, x):
    alpha, beta1 = params
    F = alpha + beta1 * x[:, 0]
    return np.exp(F) / (1 + np.exp(F))


def log_likelihood(params, x, y):
    p = logistic_func(params, x)
    log_likelihood = -1 * np.sum(y * np.log(p) + (1 - y) * np.log(1 - p))
    return log_likelihood


def userdefine_logistic_regression_task(
        fit_df: pd.DataFrame,
        models_path: Path,
        model_name: str,
        y_col_name: str = "NIHL1234_Y",
        params_init: list = [-4.18049946, 0.07176393],
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
    """

    minimize_method = kwargs.pop(
        "minimize_method", "Nelder-Mead"
    )  #"SLSQP", #"Powell", #"L-BFGS-B", #"Nelder-Mead", #"BFGS",
    minimize_options = kwargs.pop("minimize_options", {'maxiter': 10000})
    minimize_bounds = kwargs.pop("minimize_bounds", None)

    work_df = fit_df.copy()

    y = work_df[y_col_name]
    X = work_df.drop(columns=[y_col_name])
    logger.info(f"minimize method: {minimize_method}")
    logger.info(f"minimize bounds: {minimize_bounds}")
    results = minimize(log_likelihood,
                       params_init,
                       args=(X.values, y.values),
                       method=minimize_method,
                       options=minimize_options,
                       bounds=minimize_bounds)
    logger.info(f"Fit status: {results.success}")
    logger.info(f"Log likehood: {round(results.fun,2)}")
    logger.info(f"Iterations: {results.nit}")
    best_params_estimated = results.x
    best_log_likelihood_value = results.fun

    logger.info(
        f"Final result: {seq(best_params_estimated).map(lambda x: round(x,4))}. \n Log likelihood: {round(best_log_likelihood_value,2)}"
    )

    pickle.dump([
        best_params_estimated,
        best_log_likelihood_value
    ], open(models_path / Path(y_col_name + "-" + model_name), "wb"))
    return best_params_estimated, best_log_likelihood_value


def userdefine_logistic_regression_plot(
        best_params_estimated,
        age: np.array = np.arange(15, 70),
        **kwargs):
    """_summary_
    概率曲线绘制函数

    Args:
        best_params_estimated (_type_): _description_
        age (np.array, optional): _description_. Defaults to np.arange(15, 70).

    Raises:
        ValueError: _description_
    """
    key_point_x = kwargs.pop("key_point_x", 65)
    
    logger.info(f"Params: {best_params_estimated}")

    plot_X = age[:, np.newaxis]
    pred_y = logistic_func(x=plot_X, params=best_params_estimated)
    fig, ax = plt.subplots(1, figsize=(6.5, 5))
    ax.plot(plot_X, pred_y, alpha=0.4)
    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()
    key_point_y = logistic_func(x=np.array([[key_point_x]]), params=best_params_estimated)[0]
    ax.vlines(x=key_point_x, ymin=y_min, ymax=key_point_y, colors="black", linestyles=":")
    ax.hlines(y=key_point_y, xmin=x_min, xmax=key_point_x, colors="black", linestyles=":")
    ax.annotate(f"key point: ({key_point_x},{round(key_point_y,2)})",
                xy=(key_point_x, key_point_y),
                xytext=(key_point_x - (x_max-x_min)/5, key_point_y - (y_max-y_min)/10),
                color="red",
                arrowprops=dict(color="red", arrowstyle="->"))
    ax.set_ylabel("Base Risk of NIHL with age")
    ax.set_xlabel("Age (yrs)")
    plt.show()


if __name__ == "__main__":
    from datetime import datetime
    logger.add(
        f"./log/Chinese_control_group_logistic_regression_0-{datetime.now().strftime('%Y-%m-%d')}.log",
        level="INFO")

    import argparse
    parser = argparse.ArgumentParser()
    # parser.add_argument("--input_paths",
    #                     type=list,
    #                     default=[
    #                         "./cache/extract_Chinese_data.pkl",
    #                         "./cache/extract_Chinese_control_data.pkl"
    #                     ])
    # parser.add_argument("--task", type=str, default="extract")
    parser.add_argument(
        "--input_paths",
        type=str,
        default="./cache/Chinese_extract_control_classifier_df.csv")
    parser.add_argument("--task", type=str, default="analysis")
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

    if isinstance(args.input_paths, list):
        input_paths = seq(args.input_paths).map(lambda x: Path(x)).list()
    else:
        input_paths = Path(args.input_paths)
    output_path = Path(args.output_path)
    models_path = Path(args.models_path)
    additional_set = args.additional_set
    annotated_bad_case = args.annotated_bad_case
    task = args.task
    n_jobs = args.n_jobs
    for out_path in (output_path, models_path):
        if not out_path.exists():
            out_path.mkdir(parents=True)

    if task == "extract":
        final_filter_df = pd.DataFrame()
        for input_path in input_paths:
            original_data = pickle.load(open(input_path, "rb"))
            if "control" in input_path.stem:
                original_data = seq(original_data).list()
                extract_df = extract_data_for_task(df=original_data,
                                                   n_jobs=n_jobs,
                                                   **additional_set)
                filter_df = filter_data(
                    df_total=extract_df,
                    drop_col=None,
                    dropna_set=["NIHL1234"],
                    str_filter_dict={"staff_id": annotated_bad_case},
                    num_filter_dict={
                        "age": {
                            "up_limit": 60,
                            "down_limit": 15
                        },
                    },
                    eval_set=None)
            else:
                original_data = seq(original_data).flatten().list()
                extract_df = extract_data_for_task(df=original_data,
                                                   n_jobs=n_jobs,
                                                   **additional_set)
                filter_df = filter_data(
                    df_total=extract_df,
                    drop_col=None,
                    dropna_set=["NIHL1234", "LAeq"],
                    str_filter_dict={"staff_id": annotated_bad_case},
                    num_filter_dict={
                        "age": {
                            "up_limit": 60,
                            "down_limit": 15
                        },
                        "LAeq": {
                            "up_limit": 70,
                            "down_limit": 0
                        },
                    },
                    eval_set=None)

            filter_df.index = filter_df.staff_id
            filter_df.drop("staff_id", axis=1, inplace=True)
            final_filter_df = pd.concat([final_filter_df, filter_df], axis=0)
        final_filter_df.to_csv(output_path /
                         "Chinese_extract_control_classifier_df.csv",
                         header=True,
                         index=True)
    if task == "analysis":
        extract_df = pd.read_csv(input_paths, header=0, index_col="staff_id")

        # 使用全部数据
        ## NIHL1234_Y
        fit_df = extract_df[["age", "NIHL1234_Y"]]
        userdefine_logistic_regression_task(
            fit_df=fit_df,
            models_path=models_path,
            model_name="Chinese_control_group_udlr_model_0.pkl",
            y_col_name="NIHL1234_Y",
            params_init=[-3, 0.08],
        )
        best_params_estimated, best_log_likelihood_value = pickle.load(
            open(
                models_path /
                Path("NIHL1234_Y-Chinese_control_group_udlr_model_0.pkl"),
                "rb"))
        userdefine_logistic_regression_plot(
            best_params_estimated=best_params_estimated,
            age=np.arange(15, 70)
        )
        
    print(1)
