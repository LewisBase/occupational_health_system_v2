# -*- coding: utf-8 -*-
"""
@DATE: 2024-01-02 11:15:29
@Author: Liu Hengjiang
@File: examples\F_weighting_studying-12_29\comparasion_between_AC.py
@Software: vscode
@Description:
        对比AC计权在NIPTS上的表现
"""

import re
import pickle
import pandas as pd
import numpy as np
import statsmodels.api as sm
import pickle
from joblib import Parallel, delayed
from pathlib import Path
from functional import seq
from itertools import product
from loguru import logger
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures, MinMaxScaler
from sklearn.metrics import mean_absolute_error
from catboost import CatBoostRegressor

from staff_info import StaffInfo
from diagnose_info.auditory_diagnose import AuditoryDiagnose
from utils.plot_helper import plot_corr_hotmap, plot_feature_importance

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

general_calculate_func = {
    "arimean": np.mean,
    "median": np.median,
    "geomean": lambda x: 10**(np.mean(np.log10(x))),
}

catboost_regressor_params = {
    "learning_rate": 0.05,
    "l2_leaf_reg": 3,
    "max_depth": 10,
    "n_estimators": 5000,
    "early_stopping_rounds": 1000,
    "eval_metric": "RMSE",
    "metric_period": 50,
    "od_type": "Iter",
    "loss_function": "RMSE",
    "verbose": 1000,
    "random_seed": 42,
    "task_type": "CPU"
}


def statsmodels_OLS_fit(df_box, regression_X_col, regression_y_col):
    tasks = [f"{regression_y_col} ~ {regression_X_col}"]
    models = []
    for task in tasks:
        model = sm.formula.ols(task, data=df_box).fit()
        logger.info(f"{model.summary()}")
        models.append(model)
    return models


def _extract_data_for_task(data: StaffInfo, **additional_set):
    better_ear_strategy = additional_set.pop("better_ear_strategy")
    NIPTS_diagnose_strategy = additional_set.pop("NIPTS_diagnose_strategy")

    res = {}
    res["staff_id"] = data.staff_id
    # label information
    res["NIPTS"] = data.staff_health_info.auditory_diagnose.get("NIPTS")
    for freq in [1000, 2000, 3000, 4000, 6000]:
        res["NIPTS_" + str(freq)] = AuditoryDiagnose.NIPTS(
            detection_result=data.staff_health_info.auditory_detection["PTA"],
            sex=data.staff_basic_info.sex,
            age=data.staff_basic_info.age,
            mean_key=[freq],
            NIPTS_diagnose_strategy=NIPTS_diagnose_strategy)
    # feature information
    ## L
    res["Leq"] = data.staff_occupational_hazard_info.noise_hazard_info.Leq
    res["LAeq"] = data.staff_occupational_hazard_info.noise_hazard_info.LAeq
    res["LCeq"] = data.staff_occupational_hazard_info.noise_hazard_info.LCeq
    ## I
    res["Ieq"] = 10**(
        data.staff_occupational_hazard_info.noise_hazard_info.Leq / 10)
    res["IAeq"] = 10**(
        data.staff_occupational_hazard_info.noise_hazard_info.LAeq / 10)
    res["ICeq"] = 10**(
        data.staff_occupational_hazard_info.noise_hazard_info.LCeq / 10)
    ## adjust L
    for method, algorithm_code in product(
        ["total_ari", "total_geo", "segment_ari"],
        ["A+n", "A+A", "C+n", "C+C"]):
        res[f"L{algorithm_code[0]}eq_adjust_{method}_{algorithm_code}"] = data.staff_occupational_hazard_info.noise_hazard_info.L_adjust[
            method].get(algorithm_code)
    ## kurtosis
    res["kurtosis_geomean"] = data.staff_occupational_hazard_info.noise_hazard_info.kurtosis_geomean
    res["A_kurtosis_geomean"] = data.staff_occupational_hazard_info.noise_hazard_info.A_kurtosis_geomean
    res["C_kurtosis_geomean"] = data.staff_occupational_hazard_info.noise_hazard_info.C_kurtosis_geomean
    res["log-kurtosis_geomean"] = np.log10(
        data.staff_occupational_hazard_info.noise_hazard_info.kurtosis_geomean)
    res["log-A_kurtosis_geomean"] = np.log10(
        data.staff_occupational_hazard_info.noise_hazard_info.
        A_kurtosis_geomean)
    res["log-C_kurtosis_geomean"] = np.log10(
        data.staff_occupational_hazard_info.noise_hazard_info.
        C_kurtosis_geomean)
    res["backwards-kurtosis_geomean"] = (
        data.staff_occupational_hazard_info.noise_hazard_info.kurtosis_geomean
    )**-1
    res["backwards-A_kurtosis_geomean"] = (
        data.staff_occupational_hazard_info.noise_hazard_info.
        A_kurtosis_geomean)**-1
    res["backwards-C_kurtosis_geomean"] = (
        data.staff_occupational_hazard_info.noise_hazard_info.
        C_kurtosis_geomean)**-1
    ## Peak SPL
    res["max_Peak_SPL_dB"] = data.staff_occupational_hazard_info.noise_hazard_info.Max_Peak_SPL_dB
    ## other features in frequency domain
    for key, value in data.staff_occupational_hazard_info.noise_hazard_info.parameters_from_file.items(
    ):
        if (re.findall(r"\d+",
                       key.split("_")[1])
                if len(key.split("_")) > 1 else False):
            if key.split("_")[0] != "Leq":
                for func_name, func in general_calculate_func.items():
                    res[key + "_" + func_name] = func(value)
            else:
                res[key] = value

    return res


def extract_data_for_task(df, n_jobs=-1, **additional_set):
    res = Parallel(n_jobs=n_jobs, backend="multiprocessing")(
        delayed(_extract_data_for_task)(data=data, **additional_set)
        for data in df)
    res_df = pd.DataFrame(res)
    return res_df


def get_importance_from_regressor(X: pd.DataFrame,
                                  y: pd.Series,
                                  catboost_regressor_params: dict,
                                  poly_degree: int = 1,
                                  **kwargs):
    poly = PolynomialFeatures(degree=poly_degree,
                              include_bias=False,
                              interaction_only=True)
    X = poly.fit_transform(X)
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X,
                                                        y,
                                                        test_size=0.2,
                                                        random_state=42)
    model = CatBoostRegressor(**catboost_regressor_params)
    model.fit(X_train, y_train)
    y_predict = model.predict(X_test)
    MAE = mean_absolute_error(y_test, y_predict)
    logger.info(f"ALL feature for model {y.name}: MAE = {round(MAE,4)}")

    importances = model.get_feature_importance()
    featurenames = poly.get_feature_names_out()
    importances_res = dict(zip(featurenames, importances))
    return model, MAE, importances_res


if __name__ == "__main__":
    from datetime import datetime
    logger.add(
        f"./log/comparasion_between_AC-{datetime.now().strftime('%Y-%m-%d')}.log",
        level="INFO")

    import argparse
    parser = argparse.ArgumentParser()
    # parser.add_argument("--task", type=str, default="extract")
    # parser.add_argument("--input_path",
    #                     type=str,
    #                     default="./cache/extract_data.pkl")
    parser.add_argument("--task", type=str, default="analysis")
    parser.add_argument("--input_path",
                        type=str,
                        default="./results/extract_df.csv")
    parser.add_argument("--output_path", type=str, default="./results")
    parser.add_argument("--picture_path", type=str, default="./pictures")
    parser.add_argument("--model_path", type=str, default="./models")
    parser.add_argument("--additional_set",
                        type=dict,
                        default={
                            "mean_key": [3000, 4000, 6000],
                            "better_ear_strategy": "optimum",
                            "NIPTS_diagnose_strategy": "better"
                        })
    parser.add_argument("--n_jobs", type=int, default=-1)
    parser.add_argument("--is_show", type=bool, default=False)
    args = parser.parse_args()

    logger.info("Input Parameters informations:")
    message = "\n".join([f"{k:<20}: {v}" for k, v in vars(args).items()])
    logger.info(message)

    task = args.task
    input_path = Path(args.input_path)
    output_path = Path(args.output_path)
    picture_path = Path(args.picture_path)
    model_path = Path(args.model_path)
    additional_set = args.additional_set
    n_jobs = args.n_jobs
    is_show = args.is_show

    for output in (output_path, picture_path, model_path):
        if not output.exists():
            output.mkdir(parents=True)

    if task == "extract":
        original_data = pickle.load(open(input_path, "rb"))

        extract_df = extract_data_for_task(df=original_data,
                                           n_jobs=n_jobs,
                                           **additional_set)
        extract_df.index = extract_df.staff_id
        extract_df.drop("staff_id", axis=1, inplace=True)
        extract_df.to_csv(output_path / "extract_df.csv",
                          header=True,
                          index=True)

    # for X_col in ["Leq", "LAeq", "LCeq"]:
    #     models = statsmodels_OLS_fit(df_box=extract_df,
    #                                  regression_X_col=X_col,
    #                                  regression_y_col="NIPTS")

    if task == "analysis":
        extract_df = pd.read_csv(input_path, header=0, index_col="staff_id")
        label_columns = seq(
            extract_df.columns).filter(lambda x: x.startswith("NIPTS")).list()
        feature_columns = seq(extract_df.columns).filter(
            lambda x: not x.startswith("NIPTS")).list()
        # corr matrix hotmap
        for i in range(0, len(feature_columns), 10 - len(label_columns)):
            plot_df = extract_df[label_columns +
                                 feature_columns[i:i + 10 -
                                                 len(label_columns)]]
            plot_corr_hotmap(df=plot_df,
                             output_path=picture_path,
                             picture_name=f"corr_hotmap_{i}.png",
                             is_show=is_show)

        # machine learning feature extract
        X = extract_df.drop(label_columns, axis=1).copy()
        for col in label_columns:
            y = extract_df[col].copy()
            model, importances_res = get_importance_from_regressor(
                X=X,
                y=y,
                catboost_regressor_params=catboost_regressor_params,
                poly_degree=1)
            pickle.dump(model, open(model_path / f"regression_model_for_{y.name}.pkl", "wb"))
            pickle.dump(importances_res, open(model_path / f"feature_importance_for_{y.name}.pkl", "wb"))

            plot_feature_importance(
                feature_importance=importances_res,
                top_n=15,
                output_path=picture_path,
                is_show=is_show,
                picture_name=f"{y.name}-feature_importances.png")
    print(1)
