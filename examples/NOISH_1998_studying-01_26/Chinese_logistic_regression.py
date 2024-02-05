# -*- coding: utf-8 -*-
"""
@DATE: 2024-01-31 09:51:22
@Author: Liu Hengjiang
@File: examples\\NOISH_1998_studying-01_26\Chinese_logistic_regression.py
@Software: vscode
@Description:
        针对收集的国内工人工厂噪声暴露的数据进行二分类的逻辑回归尝试
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
    res = {}
    res["staff_id"] = data.staff_id
    # worker information
    res["age"] = data.staff_basic_info.age
    res["duration"] = data.staff_basic_info.duration

    # worker health infomation
    res["HL1234"] = data.staff_health_info.auditory_detection.get("PTA").mean(
        mean_key=[1000, 2000, 3000, 4000])

    res["HL1234_Y"] = 0 if res["HL1234"] <= 25 else 1

    # noise information
    res["LAeq"] = data.staff_occupational_hazard_info.noise_hazard_info.LAeq
    res["LAeq_adjust_geomean"] = data.staff_occupational_hazard_info.noise_hazard_info.L_adjust[
        "total_geo"]["A+n"]

    return res


def extract_data_for_task(df, n_jobs=-1, **additional_set):
    res = Parallel(n_jobs=n_jobs, backend="multiprocessing")(
        delayed(_extract_data_for_task)(data=data, **additional_set)
        for data in df)
    res = pd.DataFrame(res)
    return res


def logistic_func(params, x):
    alpha, beta1, beta21, beta22, beta23, phi = params
    F = alpha + beta1 * x[:, 1] + beta21 * np.power(
        x[:, 3] * (x[:, 2]), phi) + beta22 * np.power(
            x[:, 4] *
            (x[:, 2]), phi) + beta23 * np.power(x[:, 5] * (x[:, 2]), phi)
    return np.exp(F) / (1 + np.exp(F))


def log_likelihood(params, x, y):
    p = logistic_func(params, x)
    log_likelihood = -1 * np.sum(y * np.log(p) + (1 - y) * np.log(1 - p))
    return log_likelihood


def statsmodels_logistic_fit(df_box, regression_X_col, regression_y_col):
    y = df_box[regression_y_col]
    X = df_box[regression_X_col]
    X = sm.add_constant(X)
    model = sm.Logit(y, X).fit()
    logger.info(f"{model.summary()}")
    return model


if __name__ == "__main__":
    from datetime import datetime
    logger.add(
        f"./log/Chinese_logistic_regression-{datetime.now().strftime('%Y-%m-%d')}.log",
        level="INFO")

    import argparse
    parser = argparse.ArgumentParser()
    # parser.add_argument("--input_path",
    #                     type=str,
    #                     default="./cache/extract_Chinese_data.pkl")
    # parser.add_argument("--task", type=str, default="extract")
    parser.add_argument("--input_path",
                        type=str,
                        default="./cache/Chinese_extract_df.csv")
    parser.add_argument("--task", type=str, default="analysis")
    parser.add_argument("--output_path", type=str, default="./cache")
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
            "沃尔夫链条-60", "杭州重汽发动机有限公司-10", "浙江红旗机械有限公司-6",
            "Wanhao furniture factory-41",
            "Songxia electrical appliance factory-40",
            "Songxia electrical appliance factory-18",
            "Songxia electrical appliance factory-15",
            "Mamibao baby carriage manufactory-77", "Liyuan hydroelectric-51",
            "Liyuan hydroelectric-135", "Liyuan hydroelectric-112",
            "Liyuan hydroelectric-103", "Huahui Machinery-11",
            "Hebang brake pad manufactory-95",
            "Hebang brake pad manufactory-94", "Gujia furniture factory-9",
            "Gujia furniture factory-85", "Gujia furniture factory-54",
            "Gujia furniture factory-5", "Gujia furniture factory-39",
            "Gujia furniture factory-35",
            "Gengde electronic equipment factory-57",
            "Gengde electronic equipment factory-47",
            "Changhua Auto Parts Manufactory-6",
            "Changhua Auto Parts Manufactory-127",
            "Botai furniture manufactory-17", "Banglian spandex-123",
            "Changhua Auto Parts Manufactory-40", "Banglian spandex-12",
            "Changhua Auto Parts Manufactory-270",
            "Changhua Auto Parts Manufactory-48", "Gujia furniture factory-35",
            "Hebang brake pad manufactory-165",
            "Hebang brake pad manufactory-20", "Hengfeng paper mill-31",
            "Liyuan hydroelectric-135", "Liyuan hydroelectric-30",
            "NSK Precision Machinery Co., Ltd-109",
            "NSK Precision Machinery Co., Ltd-345",
            "Songxia electrical appliance factory-15",
            "Waigaoqiao Shipyard-170", "Waigaoqiao Shipyard-94", "春风动力-119",
            "浙江红旗机械有限公司-20", "浙江红旗机械有限公司-5", "Banglian spandex-123",
            "Botai furniture manufactory-66",
            "Changhua Auto Parts Manufactory-120",
            "Changhua Auto Parts Manufactory-141",
            "Changhua Auto Parts Manufactory-355",
            "Changhua Auto Parts Manufactory-40", "Gujia furniture factory-39",
            "Gujia furniture factory-5", "Gujia furniture factory-85",
            "Hengfeng paper mill-27", "Hengjiu Machinery-15",
            "Liyuan hydroelectric-120", "Liyuan hydroelectric-14",
            "NSK Precision Machinery Co., Ltd-288",
            "NSK Precision Machinery Co., Ltd-34", "Yufeng paper mill-26",
            "春风动力-98", "春江-1", "东华链条厂-60", "东华链条厂-77", "东华链条厂-79", "双子机械-9",
            "沃尔夫链条-59", "中国重汽杭州动力-83", "Wanhao furniture factory-24",
            "永创智能-46", "Wanhao furniture factory-34", "永创智能-45", "总装配厂-117",
            "总装配厂-467", "东风汽车有限公司商用车车身厂-259", "东风汽车紧固件有限公司-405",
            "东风汽车车轮有限公司-16", "Huahui Machinery-10", "Gujia furniture factory-3",
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
    additional_set = args.additional_set
    annotated_bad_case = args.annotated_bad_case
    task = args.task
    n_jobs = args.n_jobs
    if not output_path.exists():
        output_path.mkdir(parents=True)

    if task == "extract":
        original_data = pickle.load(open(input_path, "rb"))
        original_data = seq(original_data).flatten().list()
        extract_df = extract_data_for_task(df=original_data,
                                           n_jobs=n_jobs,
                                           **additional_set)
        filter_df = filter_data(
            df_total=extract_df,
            drop_col=None,
            dropna_set=["HL1234", "LAeq_adjust_geomean"],
            str_filter_dict={"staff_id": annotated_bad_case},
            num_filter_dict={
                "age": {
                    "up_limit": 60,
                    "down_limit": 15
                },
                "LAeq": {
                    "up_limit": 102,
                    "down_limit": 73 
                },
            },
            eval_set=None)

        filter_df.index = filter_df.staff_id
        filter_df.drop("staff_id", axis=1, inplace=True)
        filter_df.to_csv(output_path / "Chinese_extract_df.csv",
                         header=True,
                         index=True)
    if task == "analysis":
        extract_df = pd.read_csv(input_path, header=0, index_col="staff_id")

        # fig, ax = plt.subplots(1, figsize=(6.5, 5))
        # ax.scatter(extract_df["LAeq"], extract_df["duration"], alpha=0.4)
        # ax.set_xlabel("Sound Level in dB")
        # ax.set_ylabel("Duration Expressed (years)")
        # plt.show()
        # plt.close()

        # for Lcontrol in [60, 65, 70, 75, 79]:
        #     extract_df["Ldiff"] = (extract_df["LAeq"] - Lcontrol)**2
        #     regression_X_col = ["age", "Ldiff"]
        #     regression_y_col = ["HL1234_Y"]
        #     statsmodels_logistic_fit(df_box=extract_df,
        #                              regression_X_col=regression_X_col,
        #                              regression_y_col=regression_y_col)

        # Best model in paper
        duration_cut = [1, 4, 10, np.inf]
        extract_df["duration_box_best"] = extract_df["duration"].apply(
            lambda x: mark_group_name(x, qcut_set=duration_cut, prefix="D-"))
        fit_df = extract_df.query(
            "duration_box_best in ('D-1', 'D-2', 'D-3')")[[
                "age", "LAeq", "duration_box_best", "HL1234_Y"
            ]]
        fit_df = pd.get_dummies(fit_df, columns=["duration_box_best"])
        Lcontrol = 73
        fit_df["LAeq"] -= Lcontrol
        fit_df["LAeq"] /= fit_df["LAeq"].max()

        y = fit_df["HL1234_Y"]
        X = fit_df.drop(columns=["HL1234_Y"])
        X = sm.add_constant(X)
        params_init = 0.015 * np.ones(6)
        results = minimize(log_likelihood,
                           params_init,
                           args=(X.values, y.values),
                           method="BFGS",
                           options={'maxiter': 100000})
        params_estimated = results.x

        # plot logistic
        age = 45
        LAeq = np.arange(70, 101)
        plot_X = np.stack([
            np.ones(len(LAeq)),
            age * np.ones(len(LAeq)),
            (LAeq - 73) / (102 - 73),
            np.zeros(len(LAeq)),
            np.zeros(len(LAeq)),
            np.ones(len(LAeq)),
        ],
                          axis=1)
        pred_y = logistic_func(params=params_estimated, x=plot_X)
        fig, ax = plt.subplots(1, figsize=(6.5, 5))
        ax.plot(LAeq, pred_y, alpha=0.4)
        ax.set_title(f"Age = {age}, Duration > 10")
        ax.set_ylabel("Probability")
        ax.set_xlabel("Sound Level in dB")
        plt.show()
        plt.close()

    print(1)
