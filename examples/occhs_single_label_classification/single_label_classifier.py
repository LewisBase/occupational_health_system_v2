# -*- coding: utf-8 -*-
"""
@DATE: 2023-12-11 11:32:46
@Author: Liu Hengjiang
@File: examples\initial_test\single_label_classifier.py
@Software: vscode
@Description:
        进行上岗前体检后在岗中检测出职业病确诊概率的预测
"""

import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from functional import seq
from loguru import logger
from sklearn.model_selection import train_test_split
from sklearn import metrics
from catboost import Pool, CatBoostClassifier
from catboost.utils import get_roc_curve, get_fnr_curve, get_fpr_curve
import shap
import pickle

from utils.data_helper import get_categorical_indicies

def step(input_path, pictures_path, models_path, task, plot_types):
    input_df = pd.read_csv(input_path, header=0)
    labels = input_df["label"]
    features = input_df.drop(["id", "label"], axis=1)

    cat_features = get_categorical_indicies(features)
    features[cat_features] = features.fillna(value="", inplace=True)

    train_X, val_X, train_y, val_y = train_test_split(features,
                                                      labels,
                                                      train_size=0.8,
                                                      random_state=42)
    train_pool = Pool(train_X, train_y, cat_features=cat_features)
    eval_pool = Pool(val_X, val_y, cat_features=cat_features)

    if task == "train":
        # 训练分类模型，进行随机网格搜索
        params = {
            "loss_function": "Logloss",
            "auto_class_weights": "Balanced",  # 处理非平衡类别，其他选项有None，SqrtBalanced
            "eval_metric": "Accuracy",
            "od_type": "Iter",
            "cat_features": cat_features,
            "early_stopping_rounds": 100,
            "verbose": False,
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
                                plot=True,
                                plot_file=str(pictures_path /
                                              "randomized_search.html"))
        pickle.dump(model, open(models_path / "model.pkl", "wb"))
    else:
        model = pickle.load(open(models_path / "model.pkl", "rb"))

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
        plt.savefig(pictures_path / "ROC_fig.png")
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
        plt.savefig(pictures_path / "FPR_fig.png")
        # plt.show()
        plt.close()
    if "shap" in plot_types:
        # 绘制对特征的解释结果
        shap_values = model.get_feature_importance(train_pool,
                                                   type='ShapValues')
        # expected_value = shap_values[0,-1]
        shap_values = shap_values[:, :-1]
        shap.initjs()
        shap.summary_plot(shap_values, train_X)
        plt.savefig(pictures_path / "shap_fig.png")
        plt.close()

if __name__ == "__main__":
    from matplotlib.font_manager import FontProperties
    from matplotlib import rcParams
    
    config = {
                "font.family": "serif",
                "font.size": 12,
                "mathtext.fontset": "stix",# matplotlib渲染数学字体时使用的字体，和Times New Roman差别不大
                "font.serif": ["STZhongsong"],# 华文中宋
                "axes.unicode_minus": False # 处理负号，即-号
             }
    rcParams.update(config)
    

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path",
                        type=str,
                        default="./cache/preprocessed_data_set.csv")
    parser.add_argument("--output_path", type=str, default="./results")
    parser.add_argument("--pictures_path", type=str, default="./pictures")
    parser.add_argument("--models_path", type=str, default="./models")
    parser.add_argument("--task", type=str, default="train")
    parser.add_argument("--plot_types",
                        type=list,
                        default=["ROC", "TPR", "shap"])
    args = parser.parse_args()

    logger.info("Input Parameters informations:")
    message = "\n".join([f"{k:<20}: {v}" for k, v in vars(args).items()])
    logger.info(message)

    input_path = Path(args.input_path)
    output_path = Path(args.output_path)
    pictures_path = Path(args.pictures_path)
    models_path = Path(args.models_path)
    task = args.task
    plot_types = args.plot_types

    for path in (output_path, pictures_path, models_path):
        if not path.exists():
            path.mkdir(parents=True)

    step(input_path, pictures_path, models_path, task, plot_types)
