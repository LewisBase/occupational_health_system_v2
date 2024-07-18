# -*- coding: utf-8 -*-
"""
@DATE: 2024-07-16 11:41:56
@Author: Liu Hengjiang
@File: examples\segment_adjustment_studying-07_05\AriGeoAdjust_regression.py
@Software: vscode
@Description:
        在相同的数据集上再次验证使用均值回归拟合得到的结果
"""
import pickle
import random
import numpy as np
import pandas as pd
import scipy.stats as stats
import statsmodels.api as sm
from joblib import Parallel, delayed
from pathlib import Path
from functional import seq
from loguru import logger
from statsmodels.stats.anova import anova_lm
from scipy.optimize import curve_fit

from utils.plot_helper import plot_group_scatter, plot_emm_group_bar, plot_logistic_scatter_line
from utils.data_helper import box_data_multi

from extract_all_Chinese_data import TrainDataSet


def _extract_data_for_task(data):
    # 重新提取信息
    res = {}
    res["staff_id"] = data[2]["staff_id"]
    res["sex"] = data[2]["sex"]
    res["age"] = data[2]["age"]
    res["duration"] = data[2]["duration"]
    # label information
    res["NIPTS"] = data[1]
    # feature information
    ## L
    res["LAeq"] = data[2]["LAeq"]
    ## kurtosis
    res["kurtosis_arimean"] = data[2]["kurtosis_arimean"]
    res["kurtosis_geomean"] = data[2]["kurtosis_geomean"]
    return res


def extract_data_for_task(dataset, n_jobs=-1):
    res = Parallel(n_jobs=n_jobs, backend="multiprocessing")(
        delayed(_extract_data_for_task)(data=data) for data in dataset)
    res_df = pd.DataFrame(res)
    return res_df


def statsmodels_OLS_fit(df_box, regression_refer_col, beta_baseline):
    df_box[f"log_{regression_refer_col}"] = np.log10(
        df_box[regression_refer_col] / beta_baseline)
    # tasks = [f"NIPTS ~ LAeq + log_{regression_refer_col}"]
    tasks = ["NIPTS ~ LAeq", f"NIPTS ~ LAeq + log_{regression_refer_col}"]
    models = []
    for task in tasks:
        model = sm.formula.ols(task, data=df_box).fit()
        logger.info(f"{model.summary()}")
        models.append(model)
    logger.info(f"{anova_lm(models[0],models[1])}")


if __name__ == "__main__":
    from datetime import datetime
    logger.add(
        f"./log/AriGeoAdjust_regression-{datetime.now().strftime('%Y-%m-%d')}.log",
        level="INFO")

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, default="./results")
    parser.add_argument("--model_path", type=str, default="./models")
    parser.add_argument("--task", type=str, default="train")
    parser.add_argument("--beta_baseline", type=int, default=3)
    parser.add_argument("--qcut_sets",
                        type=list,
                        default=[[3, 10, 50, np.inf], [3, 10, 25, np.inf]])
    parser.add_argument("--groupby_cols",
                        type=list,
                        default=["kurtosis_arimean", "kurtosis_geomean"])
    parser.add_argument("--n_jobs", type=int, default=-1)
    args = parser.parse_args()

    logger.info("Input Parameters informations:")
    message = "\n".join([f"{k:<20}: {v}" for k, v in vars(args).items()])
    logger.info(message)

    input_path = Path(args.input_path)
    model_path = Path(args.model_path)
    task = args.task
    beta_baseline = args.beta_baseline
    qcut_sets = args.qcut_sets
    groupby_cols = args.groupby_cols
    n_jobs = args.n_jobs

    qcut_set_dict = dict(zip(groupby_cols, qcut_sets))
    prefixs = seq(["kurtosis_arimean",
                   "kurtosis_geomean"]).map(lambda x: x.split("_")[0][0].upper(
                   ) + x.split("_")[1][0].upper() + "-").list()

    # mission start
    ## load tensor data from pkl
    train_dataset = pickle.load(open(input_path / "train_dataset.pkl", "rb"))
    ## dataloader
    df_set = extract_data_for_task(dataset=train_dataset, n_jobs=n_jobs)

    # train model
    if task == "train":
        for groupby_col, qcut_set, prefix in zip(groupby_cols, qcut_sets,
                                                 prefixs):
            df_box = box_data_multi(df=df_set,
                                    col="LAeq",
                                    groupby_cols=[groupby_col],
                                    qcut_sets=[qcut_set],
                                    prefixs=[prefix])
            # box data regression
            logger.info(f"Regress on box data set")
            logger.info(f"Beta baseline = {beta_baseline}")
            statsmodels_OLS_fit(df_box=df_box,
                                regression_refer_col=groupby_col,
                                beta_baseline=beta_baseline)

    print(1)
