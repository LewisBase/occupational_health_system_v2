# -*- coding: utf-8 -*-
"""
@DATE: 2024-02-18 15:02:14
@Author: Liu Hengjiang
@File: examples\\NOISH_1998_studying-01_26\Chinese_control_group_classifier_prob_svn.py
@Software: vscode
@Description:
        针对中国工人对照组数据的svn分类模型概率密度曲线
        该概率将作为标准值用以计算超额风险概率
"""

import re
import ast
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
from pathlib import Path
from functional import seq
from itertools import product
from loguru import logger
from joblib import Parallel, delayed
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix
from sklearn import svm
from scipy.optimize import curve_fit

from staff_info import StaffInfo
from utils.data_helper import mark_group_name, filter_data, get_categorical_indicies

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
    res["sex"] = data.staff_basic_info.sex
    res["duration"] = data.staff_basic_info.duration

    # worker health infomation
    res["HL1234"] = data.staff_health_info.auditory_detection.get("PTA").mean(
        mean_key=[1000, 2000, 3000, 4000])

    res["HL1234_Y"] = 0 if res["HL1234"] <= 25 else 1

    return res


def extract_data_for_task(df, n_jobs=-1, **additional_set):
    res = Parallel(n_jobs=n_jobs, backend="multiprocessing")(
        delayed(_extract_data_for_task)(data=data, **additional_set)
        for data in df)
    res = pd.DataFrame(res)
    return res


def logistic_function(x, a, b, c):
    return c / (1 + np.exp(-a * (x - b)))


if __name__ == "__main__":
    from datetime import datetime
    logger.add(f"./log/Chinese_control_group_classifier_prob_svn-{datetime.now().strftime('%Y-%m-%d')}.log",level="INFO")
    
    
    import argparse
    parser = argparse.ArgumentParser()
    # parser.add_argument("--input_path",
    #                     type=str,
    #                     default="./cache/extract_Chinese_control_data.pkl")
    # parser.add_argument("--task", type=str, default="extract")
    parser.add_argument("--input_path",
                        type=str,
                        default="./cache/Chinese_extract_control_classifier_df.csv")
    parser.add_argument("--task", type=str, default="analysis-train")
    # parser.add_argument("--task", type=str, default="analysis")
    parser.add_argument("--output_path", type=str, default="./cache")
    parser.add_argument("--models_path", type=str, default="./models")
    parser.add_argument("--pictures_path", type=str, default="./pictures")
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
        default=[])
    parser.add_argument("--n_jobs", type=int, default=-1)
    parser.add_argument("--plot_types", type=list, default=["ROC", "PDF"])
    parser.add_argument("--key_feature", type=str, default="LAeq")
    # parser.add_argument("--key_feature", type=str, default="LAeq_adjust_geomean")
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
    plot_types = args.plot_types
    key_feature = args.key_feature

    for out_path in (output_path, models_path, pictures_path):
        if not out_path.exists():
            out_path.mkdir(parents=True)

    if task == "extract":
        original_data = pickle.load(open(input_path, "rb"))
        original_data = seq(original_data).list()
        extract_df = extract_data_for_task(df=original_data,
                                           n_jobs=n_jobs,
                                           **additional_set)
        filter_df = filter_data(
            df_total=extract_df,
            drop_col=None,
            dropna_set=["HL1234"],
            str_filter_dict={"staff_id": annotated_bad_case},
            num_filter_dict={
                "age": {
                    "up_limit": 60,
                    "down_limit": 15
                },
            },
            eval_set=None)

        filter_df.index = filter_df.staff_id
        filter_df.drop("staff_id", axis=1, inplace=True)
        filter_df.to_csv(output_path / "Chinese_extract_control_classifier_df.csv",
                         header=True,
                         index=True)
    if task.startswith("analysis"):
        extract_df = pd.read_csv(input_path, header=0, index_col="staff_id")

        # SVM classifier model
        age_cut = [15, 30, 45, 60]
        duration_cut = [1, 4, 10, np.inf]
        extract_df["sex"] = extract_df["sex"].map({"M":0, "F":1})
        extract_df["age_box"] = extract_df["age"].apply(
            lambda x: mark_group_name(x, qcut_set=age_cut, prefix=""))
        extract_df["duration_box"] = extract_df["duration"].apply(
            lambda x: mark_group_name(x, qcut_set=duration_cut, prefix=""))
        input_df = extract_df.query(
            "age_box in ('1','2','3') and duration_box in ('1', '2', '3')")[[
                "sex", "age_box", "duration_box", "HL1234_Y"
            ]]
        labels = input_df["HL1234_Y"]
        features = input_df.drop(["HL1234_Y"], axis=1)

        train_X, val_X, train_y, val_y = train_test_split(features,
                                                          labels,
                                                          train_size=0.8,
                                                          random_state=42)

        if task.split("-")[-1] == "train":
            # 训练分类模型，进行网格搜索
            params = {
                'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf'], 'gamma': [0.1, 0.5, 1]
            }
            model = svm.SVC(probability=True, **params)
            grid_search = GridSearchCV(model, param_grid=params, cv=5)
            grid_search.fit(train_X, train_y)
            model = grid_search.best_estimator_
            pickle.dump(model, open(models_path / f"Chinese_control_group_svn-{key_feature}_classifier_model.pkl", "wb"))
        else:
            model = pickle.load(open(models_path / f"Chinese_control_group_svn-{key_feature}_classifier_model.pkl", "rb"))

        if "ROC" in plot_types:
            # 绘制ROC曲线
            y_pred_prob = model.predict_proba(val_X)[:, 1]
            fpr, tpr, thresholds = roc_curve(val_y, y_pred_prob)
            roc_auc = roc_auc_score(val_y, y_pred_prob)

            lw = 2
            plt.plot(fpr,
                     tpr,
                     color='darkorange',
                     lw=lw,
                     label='ROC curve (area = %0.2f)' % roc_auc,
                     alpha=0.5)
            plt.plot([0, 1], [0, 1],
                     color='navy',
                     lw=lw,
                     linestyle='--',
                     alpha=0.5)
            plt.xlabel("False Postive Rate")
            plt.ylabel("True Postive Rate")
            plt.legend(loc="best")
            plt.title("ROC Curve")
            plt.savefig(pictures_path / f"ROC-{key_feature}_fig.png")
            # plt.show()
            plt.close()
        if "PDF" in plot_types:
            prob_df = pd.DataFrame()
            fig, ax = plt.subplots(1, figsize=(6.5, 5))
            for age_box, duration_box in product(("1","2","3"),("1","2","3")):
                LAeq = np.arange(70, 101)
                X = np.stack([LAeq, [age_box]*len(LAeq), [duration_box]*len(LAeq)], axis=1)
                prob = model.predict_proba(X)[:,1]
                sub_df = pd.DataFrame(X, columns=["LAeq", "age_box", "duration_box"])
                sub_df["LAeq"] = sub_df["LAeq"].astype(int)
                sub_df["prob"] = prob
                prob_df = pd.concat([prob_df, sub_df], axis=0)
                ax.scatter(LAeq, prob, label=f"{age_box}+{duration_box}")
            ax.set_ylabel("Probability")
            ax.set_xlabel("Sound Level in dB")
            plt.legend(loc="best", ncol=2)
            plt.savefig(pictures_path / f"Chinese_control_group_svn_prob-{key_feature}_fig.png")
            plt.close(fig=fig)
            prob_df.to_csv(output_path / f"Chinese_control_group_svn_prob-{key_feature}_df.csv", header=True, index=False)

            prob_df = prob_df.query("LAeq > 73")
            average_bin_value = prob_df.groupby("LAeq")["prob"].mean()
            average_bin_value.index = average_bin_value.index.astype(int)
            initial_guess = [0.01, 70, 1]
            params, _ = curve_fit(logistic_function, average_bin_value.index, average_bin_value.values, p0=initial_guess)
            a, b, c = params
            prob_fit = logistic_function(LAeq, a=a, b=b, c=c)
            fig, ax = plt.subplots(1, figsize=(6.5, 5))
            ax.scatter(average_bin_value.index, average_bin_value.values)
            ax.plot(LAeq, prob_fit, label="Probability fit")
            ax.set_ylabel("Probability")
            ax.set_xlabel("Sound Level in dB")
            plt.legend(loc="best")
            plt.savefig(pictures_path / f"Chinese_control_group_svn_average_prob-{key_feature}_fig.png")
            plt.close(fig=fig)

    print(1)