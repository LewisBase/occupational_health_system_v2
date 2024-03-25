# -*- coding: utf-8 -*-
"""
@DATE: 2024-01-29 10:26:29
@Author: Liu Hengjiang
@File: examples\\NOISH_1998_studying-01_26\\NOISH_logistic_regression.py
@Software: vscode
@Description:
        复现NOISH,1998论文中有关逻辑回归的内容
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
from utils.data_helper import mark_group_name

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


def age_box(age):
    if age < 17:
        return 0
    elif 17 <= age < 28:
        return 1
    elif 28 <= age < 36:
        return 2
    elif 36 <= age < 46:
        return 3
    elif 46 <= age < 54:
        return 4
    else:
        return 5


def duration_box(duration):
    if duration < 1:
        return 0
    elif 1 <= duration < 2:
        return 1
    elif 2 <= duration < 5:
        return 2
    elif 5 <= duration < 11:
        return 3
    elif 11 <= duration < 21:
        return 4
    else:
        return 5


def _extract_data_for_task(data, **additional_set):
    res = {}
    res["staff_id"] = data.staff_id
    # worker information
    res["age"] = data.staff_basic_info.age
    res["duration"] = data.staff_basic_info.duration
    res["age_box"] = age_box(data.staff_basic_info.age)
    res["duration_box"] = duration_box(data.staff_basic_info.duration)

    # worker health infomation
    res["HL1234"] = data.staff_health_info.auditory_detection.get("PTA").mean(
        mean_key=[1000, 2000, 3000, 4000])
    res["HL5123"] = data.staff_health_info.auditory_detection.get("PTA").mean(
        mean_key=[500, 1000, 2000, 3000])
    res["HL123"] = data.staff_health_info.auditory_detection.get("PTA").mean(
        mean_key=[1000, 2000, 3000])
    res["HL512"] = data.staff_health_info.auditory_detection.get("PTA").mean(
        mean_key=[500, 1000, 2000])
    res["HL346"] = data.staff_health_info.auditory_detection.get("PTA").mean(
        mean_key=[3000, 4000, 6000])

    res["HL1234_Y"] = 0 if res["HL1234"] <= 25 else 1
    res["HL5123_Y"] = 0 if res["HL5123"] <= 25 else 1
    res["HL123_Y"] = 0 if res["HL123"] <= 25 else 1
    res["HL512_Y"] = 0 if res["HL512"] <= 25 else 1
    res["HL346_Y"] = 0 if res["HL346"] <= 25 else 1

    # noise information
    res["LAeq"] = data.staff_occupational_hazard_info.noise_hazard_info.LAeq

    return res


def extract_data_for_task(df, n_jobs=-1, **additional_set):
    res = Parallel(n_jobs=n_jobs, backend="multiprocessing")(
        delayed(_extract_data_for_task)(data=data, **additional_set)
        for data in df)
    res = pd.DataFrame(res)
    return res


def logistic_func(params, x):
    alpha, beta1, beta21, beta22, beta23, phi= params
    F = alpha + beta1 * x[:, 0] + \
        beta21 * np.power(x[:, 1], phi) + \
        beta22 * np.power(x[:, 2], phi) + \
        beta23 * np.power(x[:, 3], phi)
    # alpha, beta1, beta21, beta22, beta23, phi, L0 = params
    # F = alpha + beta1 * x[:, 0] + \
    #     beta21 * np.power(x[:, 1] - L0, phi) + \
    #     beta22 * np.power(x[:, 2] - L0, phi) + \
    #     beta23 * np.power(x[:, 3] - L0, phi)
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
        f"./log/NOISH_logistic_regression-{datetime.now().strftime('%Y-%m-%d')}.log",
        level="INFO")

    import argparse
    parser = argparse.ArgumentParser()
    # parser.add_argument("--input_path",
    #                     type=str,
    #                     default="./cache/extract_NOISH_data-1234.pkl")
    parser.add_argument("--input_path",
                        type=str,
                        default="./cache/NOISH_extract_df.csv")
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
    parser.add_argument("--task", type=str, default="analysis")
    parser.add_argument("--n_jobs", type=int, default=-1)
    args = parser.parse_args()

    logger.info("Input Parameters informations:")
    message = "\n".join([f"{k:<20}: {v}" for k, v in vars(args).items()])
    logger.info(message)

    input_path = Path(args.input_path)
    output_path = Path(args.output_path)
    models_path = Path(args.models_path)
    additional_set = args.additional_set
    task = args.task
    n_jobs = args.n_jobs
    for out_path in (output_path, models_path):
        if not out_path.exists():
            out_path.mkdir(parents=True)

    if task == "extract":
        original_data = pickle.load(open(input_path, "rb"))
        extract_df = extract_data_for_task(df=original_data,
                                           n_jobs=n_jobs,
                                           **additional_set)
        extract_df.index = extract_df.staff_id
        extract_df.drop("staff_id", axis=1, inplace=True)
        extract_df.to_csv(output_path / "NOISH_extract_df.csv",
                          header=True,
                          index=True)
    if task == "analysis":
        extract_df = pd.read_csv(input_path, header=0, index_col="staff_id")

        # Best model in paper
        duration_cut = [1, 4, 10, np.inf]
        extract_df["duration_box_best"] = extract_df["duration"].apply(
            lambda x: mark_group_name(x, qcut_set=duration_cut, prefix="D-"))
        
        log_likelihood_value = []
        params_estimated = []
        for L_control in range(60, 80):
            fit_df = extract_df.query(
            "duration_box_best in ('D-1', 'D-2', 'D-3')")[[
                "age", "LAeq", "duration_box_best", "HL1234_Y"
            ]]
            fit_df = pd.get_dummies(fit_df, columns=["duration_box_best"])
            fit_df["LAeq"] -= L_control
            fit_df["LAeq"] /= fit_df["LAeq"].max()
            fit_df["duration_box_best_D-1"] *= fit_df["LAeq"]
            fit_df["duration_box_best_D-2"] *= fit_df["LAeq"]
            fit_df["duration_box_best_D-3"] *= fit_df["LAeq"]

            y = fit_df["HL1234_Y"]
            X = fit_df.drop(columns=["HL1234_Y", "LAeq"])
            # params_init = [-5, 0.08, 2, 3, 4, 3, 0.7]
            params_init = 0.02 * np.ones(6)
            results = minimize(log_likelihood,
                               params_init,
                               args=(X.values, y.values),
                               method="Nelder-Mead", #"BFGS",
                               options={'maxiter': 100000})
            logger.info(f"Fit result for L_control = {L_control}")
            logger.info(f"Fit status: {results.success}")
            logger.info(f"Log likehood: {results.fun}")
            logger.info(f"Iterations: {results.nit}")
            
            log_likelihood_value.append(results.fun)
            params_estimated.append(results.x)
        max_LAeq = extract_df["LAeq"].max()
        best_log_likelihood_value = np.min(log_likelihood_value)
        best_params_estimated = params_estimated[np.argmin(log_likelihood_value)]
        best_L_control = np.arange(60, 80)[np.argmin(log_likelihood_value)]
        logger.info(f"Final result: {best_params_estimated} + {best_L_control}. Log likelihood: {best_log_likelihood_value}")
        
        pickle.dump([best_params_estimated, best_L_control, max_LAeq, best_log_likelihood_value],
                    open(models_path / f"NOISH_experiment_group_udlr-_classifier_model.pkl", "wb"))

        # plot logistic
        age = 65
        LAeq = np.arange(50, 100)
        plot_X = np.stack([
            age * np.ones(len(LAeq)),
            (LAeq - best_L_control) / (max_LAeq - best_L_control) * np.zeros(len(LAeq)),
            (LAeq - best_L_control) / (max_LAeq - best_L_control) * np.zeros(len(LAeq)),
            (LAeq - best_L_control) / (max_LAeq - best_L_control) * np.ones(len(LAeq)),
        ],
                          axis=1)
        pred_y = logistic_func(params=best_params_estimated, x=plot_X)
        fig, ax = plt.subplots(1, figsize=(6.5, 5))
        ax.plot(LAeq, pred_y, alpha=0.4)
        ax.set_title(f"Age = {age}, Duration > 10")
        ax.set_ylabel("Probability")
        ax.set_xlabel("Sound Level in dB")
        plt.show()
        # plt.close()

    print(1)
