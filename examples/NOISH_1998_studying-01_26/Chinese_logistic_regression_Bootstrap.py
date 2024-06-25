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
from Chinese_control_group_logistic_regression_0 import logistic_func as logistic_func_control_0


def logistic_func(params, x):
    alpha, beta1, beta21, beta22, beta23, phi = params
    F = alpha + beta1 * x[:, 0] + beta21 * np.power(
        x[:, 1], phi) + beta22 * np.power(x[:, 2], phi) + beta23 * np.power(
            x[:, 3], phi)
    return np.exp(F) / (1 + np.exp(F))


def logistic_func_new(params, phi, x):
    alpha, beta1, beta21, beta22, beta23 = params
    F = alpha + beta1 * x[:, 0] + beta21 * np.power(
        x[:, 1], phi) + beta22 * np.power(x[:, 2], phi) + beta23 * np.power(
            x[:, 3], phi)
    return np.exp(F) / (1 + np.exp(F))


def log_likelihood(params, x, y):
    p = logistic_func(params, x)
    log_likelihood = -1 * np.sum(y * np.log(p) + (1 - y) * np.log(1 - p))
    return log_likelihood


def log_likelihood_new(params, phi, x, y):
    p = logistic_func_new(params, phi, x)
    log_likelihood = -1 * np.sum(y * np.log(p) + (1 - y) * np.log(1 - p))
    return log_likelihood


def userdefine_logistic_regression_task(
    resample_df: pd.DataFrame,
    params_init: list,
    phi_range: list,
    L_control: float,
    y_col_name: str = "NIHL1234_Y",
    **kwargs,
):
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
    work_df["LAeq"] -= L_control
    work_df["LAeq"] /= work_df["LAeq"].max()
    work_df["duration_box_best_D-1"] *= work_df["LAeq"]
    work_df["duration_box_best_D-2"] *= work_df["LAeq"]
    work_df["duration_box_best_D-3"] *= work_df["LAeq"]

    y = work_df[y_col_name]
    X = work_df.drop(columns=[y_col_name, "LAeq"])
    if phi_range:
        for phi in phi_range:
            results = minimize(log_likelihood_new,
                               params_init,
                               args=(phi, X.values, y.values),
                               method=minimize_method,
                               options=minimize_options,
                               bounds=minimize_bounds)
            log_likelihood_value.append(results.fun)
            params_estimated.append(results.x)
            phi_estimated.append(phi)
    else:
        results = minimize(log_likelihood,
                           params_init,
                           args=(X.values, y.values),
                           method=minimize_method,
                           options=minimize_options,
                           bounds=minimize_bounds)

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
    excess_risk = logistic_func(x=estimate_X, params=base_params_estimated)
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
                        n_iterations: int = 1000,
                        age: int = 30,
                        LAeq: np.array = np.arange(70, 100, 10),
                        duration: np.array = np.array([1, 0, 0])):
    estimates = []
    for i in tqdm(range(n_iterations)):
        resample_df = resample(data, replace=True, n_samples=data.shape[0])
        best_params_estimated, best_L_control, _ = userdefine_logistic_regression_task(
            resample_df=resample_df,
            params_init=base_params_estimated[:-1],
            phi_range=[base_params_estimated[-1]],
            L_control=base_L_control)
        LAeq_duration_matrix = np.tile(duration, (len(LAeq), 1)) * (
            (LAeq - best_L_control) /
            (max_LAeq - best_L_control))[:, np.newaxis]
        age_matrix = age * np.ones(len(LAeq))

        estimate_X = np.concatenate(
            (age_matrix[:, np.newaxis], LAeq_duration_matrix), axis=1)
        estimate_y = logistic_func(x=estimate_X, params=best_params_estimated)
        control_X = np.array([[age]])
        control_y = logistic_func_control_0(x=control_X,
                                            params=control_params_estimated)
        estimate_y -= control_y
        estimates.append(dict(zip(LAeq, estimate_y)))
    return pd.DataFrame(estimates)


def convert_num_to_str(res: pd.Series):
    return str(round(res["excess_risk"] * 100, 2)) + "\n(" + str(
        round(res["lower_bounds"] * 100, 2)) + " " + str(
            round(res["upper_bounds"] * 100, 2)) + ")"


if __name__ == "__main__":
    from datetime import datetime
    logger.add(
        f"./log/Chinese_logistic_regression_Bootstrap-{datetime.now().strftime('%Y-%m-%d')}.log",
        level="INFO")

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_path",
        type=str,
        default="./cache/Chinese_extract_experiment_classifier_df.csv")
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

        # data type convert
        duration_cut = [0, 4, 10, np.inf]
        extract_df["duration_box_best"] = extract_df["duration"].apply(
            lambda x: mark_group_name(x, qcut_set=duration_cut, prefix="D-"))

        # 使用全部数据
        ## NIHL1234_Y
        fit_df = extract_df.query(
            "duration_box_best in ('D-1', 'D-2', 'D-3') and LAeq >= 70")[[
                "age", "LAeq", "duration_box_best", "NIHL1234_Y"
            ]]
        base_params_estimated, base_L_control, max_LAeq, best_log_likelihood_value = pickle.load(
            open(
                models_path /
                Path("NIHL1234_Y-Chinese_experiment_group_udlr_model.pkl"),
                "rb"))
        control_params_estimated, control_log_likelihood_value = pickle.load(
            open(
                models_path /
                Path("NIHL1234_Y-Chinese_control_group_udlr_model_0.pkl"),
                "rb"))

        final_confidence_res = pd.DataFrame()
        for i, (age, duration) in enumerate((
            (30, np.array([1, 0,0])), (30, np.array([0, 1, 0])),
            (45, np.array([0, 1,0])), (45, np.array([0, 0, 1])),
            (65, np.array([0, 0, 1]))
            )):
            excess_risk_value = get_excess_risk(
                base_params_estimated=base_params_estimated,
                base_L_control=base_L_control,
                max_LAeq=max_LAeq,
                control_params_estimated=control_params_estimated,
                age=age,
                LAeq=np.arange(60, 130, 10),
                duration=duration)
            estimates = bootstrap_estimates(
                data=fit_df,
                base_params_estimated=base_params_estimated,
                base_L_control=base_L_control,
                max_LAeq=max_LAeq,
                control_params_estimated=control_params_estimated,
                n_iterations=1000,
                age=age,
                LAeq=np.arange(60, 130, 10),
                duration=duration)
            confidence_res = pd.DataFrame()
            confidence_res["excess_risk"] = excess_risk_value
            confidence_res["lower_bounds"] = estimates.apply(
                lambda x: np.percentile(x, 5))
            confidence_res["upper_bounds"] = estimates.apply(
                lambda x: np.percentile(x, 95))

            if duration[0] == 1:
                duration_desp = "1~4 years"
            elif duration[1] == 1:
                duration_desp = "5~10 years"
            elif duration[2] == 1:
                duration_desp = "> 10 years"
            else:
                raise ValueError
            content_dict = {
                "Age": age,
                "Duration": duration_desp,
                "60": convert_num_to_str(confidence_res.loc[60]),
                "70": convert_num_to_str(confidence_res.loc[70]),
                "80": convert_num_to_str(confidence_res.loc[80]),
                "90": convert_num_to_str(confidence_res.loc[90]),
                "100": convert_num_to_str(confidence_res.loc[100]),
                "110": convert_num_to_str(confidence_res.loc[110]),
                "120": convert_num_to_str(confidence_res.loc[120]),
            }
            final_confidence_res = pd.concat([final_confidence_res, pd.DataFrame(content_dict, index=[i])])
        print(1)
