# -*- coding: utf-8 -*-
"""
@DATE: 2024-07-19 11:01:25
@Author: Liu Hengjiang
@File: examples\\NOISH_1998_studying-01_26\Chinese_logistic_regression_total_data.py
@Software: vscode
@Description:
        使用全部的中国工人暴露数据进行逻辑回归
        综合文章描述与复现结果来看，应该不是使用全部数据进行逻辑回归
"""
import pickle
import pandas as pd
import numpy as np
from functional import seq
from pathlib import Path
from loguru import logger
from scipy.optimize import minimize

from utils.data_helper import mark_group_name
from Chinese_logistic_regression import \
    log_likelihood_original, log_likelihood_indepphi, userdefine_logistic_regression_plot

from matplotlib import rcParams

config = {
    "font.family": "serif",
    "font.size": 12,
    "mathtext.fontset": "stix",  # matplotlib渲染数学字体时使用的字体，和Times New Roman差别不大
    "font.serif": ["STZhongsong"],  # 华文中宋
    "axes.unicode_minus": False  # 处理负号，即-号
}
rcParams.update(config)


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
        work_df = fit_df.copy()
        work_df = pd.get_dummies(work_df, columns=["duration_box_best"])
        work_df["LAeq"] = work_df["LAeq"].apply(lambda x: (x - L_control) / (
            max_LAeq - L_control) if x >= 70 else 0)
        work_df["duration_box_best_D-1"] *= work_df["LAeq"]
        work_df["duration_box_best_D-2"] *= work_df["LAeq"]
        work_df["duration_box_best_D-3"] *= work_df["LAeq"]

        y = work_df[y_col_name]
        X = work_df.drop(columns=[y_col_name, "LAeq"])
        if phi_range:
            for phi in phi_range:
                logger.info(f"Fixed phi: {phi}")
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
        logger.info(f"Fit result for L_control = {L_control}")
        logger.info(f"Fit status: {results.success}")
        logger.info(f"Log likehood: {round(results.fun,2)}")
        logger.info(f"Iterations: {results.nit}")
        logger.info(f"Fit parameters: {results.x}")

    best_log_likelihood_value = np.min(log_likelihood_value)
    best_params_estimated = params_estimated[np.nanargmin(log_likelihood_value)]
    if phi_range:
        best_phi_estimated = phi_estimated[np.nanargmin(log_likelihood_value)]
        best_params_estimated = np.append(best_params_estimated,
                                          best_phi_estimated)
    best_L_control = L_control_range[np.nanargmin(log_likelihood_value)]
    logger.info(
        f"Final result: {seq(best_params_estimated).map(lambda x: round(x,4))} + {best_L_control}. \n Log likelihood: {round(best_log_likelihood_value,2)}"
    )

    pickle.dump([
        best_params_estimated, best_L_control, max_LAeq,
        best_log_likelihood_value
    ], open(models_path / Path(y_col_name + "-" + model_name), "wb"))
    return best_params_estimated, best_L_control, max_LAeq, best_log_likelihood_value


if __name__ == "__main__":
    from datetime import datetime
    logger.add(f"./log/Chinese_logistic_regression_total_data-{datetime.now().strftime('%Y-%m-%d')}.log",level="INFO")
    

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_paths",
                        type=list,
                        default=[
                            "./cache/Chinese_extract_control_classifier_df.csv",
                            "./cache/Chinese_extract_experiment_classifier_df.csv"
                        ])
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

    input_paths = seq(args.input_paths).map(lambda x: Path(x))
    output_path = Path(args.output_path)
    models_path = Path(args.models_path)
    additional_set = args.additional_set
    task = args.task
    n_jobs = args.n_jobs
    for out_path in (output_path, models_path):
        if not out_path.exists():
            out_path.mkdir(parents=True)

    if task == "analysis":
        extract_df = pd.DataFrame()
        for input_path in input_paths:
            sub_extract_df = pd.read_csv(input_path,
                                         header=0,
                                         index_col="staff_id")
            if "kurtosis_arimean" not in sub_extract_df.columns:
                sub_extract_df["kurtosis_arimean"] = 0
                sub_extract_df["kurtosis_geomean"] = 0
            else:
                sub_extract_df = sub_extract_df.query("LAeq >= 70")
            extract_df = pd.concat([extract_df, sub_extract_df], axis=0)
        extract_df.to_csv(output_path / "Chinese_extract_total_classifier_df.csv",
                          header=True,
                          index=True)

        duration_cut = [0, 4, 10, np.inf]
        extract_df["duration_box_best"] = extract_df["duration"].apply(
            lambda x: mark_group_name(x, qcut_set=duration_cut, prefix="D-"))
        max_LAeq = extract_df["LAeq"].max()

        # 使用全部对照组数据
        ## HL1234
        fit_df = extract_df.query(
            "duration_box_best in ('D-1', 'D-2', 'D-3')")[[
                "age", "LAeq", "duration_box_best", "NIHL1234_Y"
            ]]
        best_params_estimated, best_L_control, max_LAeq, best_log_likelihood_value\
            = userdefine_logistic_regression_task(
            fit_df=fit_df,
            max_LAeq=max_LAeq,
            models_path=models_path,
            model_name="Chinese_total_group_udlr_model.pkl",
            y_col_name="NIHL1234_Y",
            params_init=[-5.05, 0.08, 2.66, 3.98, 6.42, 3.4],
            L_control_range=np.arange(55, 71),)
            # phi_range=[1, 2, 3],
            # minimize_bounds = ([None,None],[None,None],[1,4],[4,6],[6,9]))

        ## plot result
        userdefine_logistic_regression_plot(
            best_params_estimated=best_params_estimated,
            best_L_control=best_L_control,
            max_LAeq=max_LAeq,
            age=30,
            duration=np.array([0, 1, 0]),
            LAeq=np.arange(60, 120),
            point_type="2nd",
            control_params_estimated=best_params_estimated[:2])

        ## HL346
        fit_df = extract_df.query(
            "duration_box_best in ('D-1', 'D-2', 'D-3')")[[
                "age", "LAeq", "duration_box_best", "NIHL346_Y"
            ]]
        best_params_estimated, best_L_control, max_LAeq, best_log_likelihood_value\
            = userdefine_logistic_regression_task(
            fit_df=fit_df,
            max_LAeq=max_LAeq,
            models_path=models_path,
            model_name="Chinese_total_group_udlr_model.pkl",
            y_col_name="NIHL346_Y",
            params_init=[-5.05, 0.08, 2.66, 3.98, 6.42, 3.4],
            L_control_range=np.arange(55, 71))

        ## plot result
        userdefine_logistic_regression_plot(
            best_params_estimated=best_params_estimated,
            best_L_control=best_L_control,
            max_LAeq=max_LAeq,
            age=45,
            duration=np.array([0, 0, 1]),
            LAeq=np.arange(60, 120),
            point_type="2nd",
            control_params_estimated=best_params_estimated[:2])

    print(1)