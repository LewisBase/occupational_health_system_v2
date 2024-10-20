# -*- coding: utf-8 -*-
"""
@DATE: 2024-03-12 14:25:51
@Author: Liu Hengjiang
@File: examples\\NOISH_1998_studying-01_26\Chinese_experiment_group_classifier_prob_catboost.py
@Software: vscode
@Description:
        针对中国工人数据进行二分类尝试，获得NIHL的概率密度曲线
        从概率密度曲线中确立噪声等效声压级保护的临界值
        
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
from sklearn.model_selection import train_test_split
from sklearn import metrics
from scipy.optimize import curve_fit
from catboost import Pool, CatBoostClassifier
from catboost.utils import get_roc_curve, get_fnr_curve, get_fpr_curve

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
    res["sex"] = data.staff_basic_info.sex
    res["age"] = data.staff_basic_info.age
    res["duration"] = data.staff_basic_info.duration

    # worker health infomation
    res["HL1234"] = data.staff_health_info.auditory_detection.get("PTA").mean(
        mean_key=[1000, 2000, 3000, 4000])

    res["HL1234_Y"] = 0 if res["HL1234"] <= 25 else 1

    # noise information
    res["LAeq"] = data.staff_occupational_hazard_info.noise_hazard_info.LAeq

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
    logger.add(f"./log/Chinese_experiment_group_classifier_prob_catboost-{datetime.now().strftime('%Y-%m-%d')}.log",level="INFO")
    
    import argparse
    parser = argparse.ArgumentParser()
    # parser.add_argument("--input_path",
    #                     type=str,
    #                     default="./cache/extract_Chinese_data.pkl")
    # parser.add_argument("--task", type=str, default="extract")
    parser.add_argument("--input_path",
                        type=str,
                        default="./cache/Chinese_extract_classifier_df.csv")
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
    parser.add_argument("--plot_types", type=list, default=["ROC"])
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
        original_data = seq(original_data).flatten().list()
        extract_df = extract_data_for_task(df=original_data,
                                           n_jobs=n_jobs,
                                           **additional_set)
        filter_df = filter_data(
            df_total=extract_df,
            drop_col=None,
            dropna_set=["HL1234", "LAeq"],
            str_filter_dict={"staff_id": annotated_bad_case},
            num_filter_dict={
                "age": {
                    "up_limit": 60,
                    "down_limit": 15
                },
                # "LAeq": {
                #     "up_limit": 100,
                #     "down_limit": 70
                # },
            },
            eval_set=None)

        filter_df.index = filter_df.staff_id
        filter_df.drop("staff_id", axis=1, inplace=True)
        filter_df.to_csv(output_path / "Chinese_extract_classifier_df.csv",
                         header=True,
                         index=True)
    if task.startswith("analysis"):
        extract_df = pd.read_csv(input_path, header=0, index_col="staff_id")

        # Catboost classifier model
        age_cut = [15, 30, 40, 60]
        duration_cut = [1, 4, 10, np.inf]
        # age_cut = [15, 30, 45, 60]
        # duration_cut = [0, 4, 10, np.inf]
        extract_df["age_box"] = extract_df["age"].apply(
            lambda x: mark_group_name(x, qcut_set=age_cut, prefix="A-"))
        extract_df["duration_box"] = extract_df["duration"].apply(
            lambda x: mark_group_name(x, qcut_set=duration_cut, prefix="D-"))
        input_df = extract_df.query(
            "duration_box in ('D-1', 'D-2', 'D-3')")[[
                key_feature, "age", "duration_box", "HL1234_Y"
            ]]
        labels = input_df["HL1234_Y"]
        features = input_df.drop(["HL1234_Y"], axis=1)

        cat_features = get_categorical_indicies(features)

        train_X, val_X, train_y, val_y = train_test_split(features,
                                                          labels,
                                                          train_size=0.8,
                                                          random_state=42)
        train_pool = Pool(train_X, train_y, cat_features=cat_features)
        eval_pool = Pool(val_X, val_y, cat_features=cat_features)

        if task.split("-")[-1] == "train":
            # 训练分类模型，进行随机网格搜索
            params = {
                "loss_function": "Logloss",
                "auto_class_weights": "SqrtBalanced",  # 处理非平衡类别，其他选项有None，SqrtBalanced
                "eval_metric": "Accuracy",
                "od_type": "Iter",
                "cat_features": cat_features,
                "early_stopping_rounds": 100,
                "verbose": 50,
                # "task_type": "GPU"
                "task_type": "CPU"
            }
            model = CatBoostClassifier(**params)
            grid = {
                "learning_rate": [0.03, 0.1],
                "depth": [4, 6, 10],
                "l2_leaf_reg": [1, 3, 5, 7, 9]
            }
            model.randomized_search(grid,
                                    train_pool,
                                    verbose=50,
                                    cv=10,
                                    )
            pickle.dump(model, open(models_path / f"Chinese_experiment_group_catboost-{key_feature}_classifier_model.pkl", "wb"))
        else:
            model = pickle.load(open(models_path / f"Chinese_experiment_group_catboost-{key_feature}_classifier_model.pkl", "rb"))

        if "ROC" in plot_types:
            # 绘制ROC曲线
            curve = get_roc_curve(model, eval_pool)
            (fpr, tpr, thresholds) = curve
            roc_auc = metrics.auc(fpr, tpr)

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
            plt.savefig(pictures_path / f"Chinese_experiment_group_catboost-{key_feature}_ROC_fig.png")
            # plt.show()
            plt.close()
        if "TPR" in plot_types:
            # 绘制FPR、FNR曲线
            (thresholds, fpr) = get_fpr_curve(curve=curve)
            (thresholds, fnr) = get_fnr_curve(curve=curve)
            lw = 2
            plt.plot(thresholds, fpr, color='blue', lw=lw, label='FPR', alpha=0.5)
            plt.plot(thresholds, fnr, color='green', lw=lw, label='FNR', alpha=0.5)
            plt.xlabel("Threshold")
            plt.ylabel("Error Rate")
            plt.legend(loc="best")
            plt.title("FPR/FNR Curve")
            plt.savefig(pictures_path / f"Chinese_experiment_group_catboost-{key_feature}_TPR_fig.png")
            # plt.show()
            plt.close()
        if "PDF" in plot_types:
            prob_df = pd.DataFrame()
            fig, ax = plt.subplots(1, figsize=(6.5, 5))
            for sex, age_box, duration_box in product(("M", "F"),("A-1","A-2","A-3"),("D-1","D-2","D-3")):
                LAeq = np.arange(70, 101)
                X = np.stack([LAeq, [sex]*len(LAeq), [age_box]*len(LAeq), [duration_box]*len(LAeq)], axis=1)
                prob = model.predict_proba(X)[:,1]
                sub_df = pd.DataFrame(X, columns=["LAeq", "sex", "age_box", "duration_box"])
                sub_df["LAeq"] = sub_df["LAeq"].astype(int)
                sub_df["prob"] = prob
                prob_df = pd.concat([prob_df, sub_df], axis=0)
                ax.scatter(LAeq, prob, label=f"{sex}+{age_box}+{duration_box}")
            ax.set_ylabel("Probability")
            ax.set_xlabel("Sound Level in dB")
            plt.legend(loc="best", ncol=2)
            plt.savefig(pictures_path / f"Chinese_experiment_group_catboost_prob-{key_feature}_fig.png")
            plt.close(fig=fig)
            prob_df.to_csv(output_path / f"Chinese_experiment_group_catboost_prob-{key_feature}_df.csv", header=True, index=False)

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
            plt.savefig(pictures_path / f"Chinese_experiment_group_catboost_average_prob-{key_feature}_fig.png")
            plt.close(fig=fig)

    print(1)