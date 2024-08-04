# -*- coding: utf-8 -*-
"""
@DATE: 2024-04-12 16:03:52
@Author: Liu Hengjiang
@File: examples\occhs_multi_label_classification\multi_label_classifier.py
@Software: vscode
@Description:
        进行根据体检结果检测出不同职业病概率的预测
"""

import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from functional import seq
from loguru import logger
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn import metrics
from catboost import Pool, CatBoostClassifier
from catboost.utils import get_roc_curve, get_fnr_curve, get_fpr_curve
import shap
import pickle

from utils.data_helper import get_categorical_indicies

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


def step(input_path, pictures_path, models_path, task):
    input_df = pd.read_csv(input_path, header=0)
    logger.info(f"Encoding label")
    mlb = MultiLabelBinarizer()
    labels = mlb.fit_transform(input_df["labels"].str.split(","))
    features = input_df.drop(["id", "labels"], axis=1)

    cat_features = get_categorical_indicies(features)
    features.iloc[:,cat_features] = features.iloc[:,cat_features].astype(str)
    features.iloc[:,cat_features].fillna(value="", inplace=True)

    train_X, val_X, train_y, val_y = train_test_split(features,
                                                      labels,
                                                      train_size=0.5,
                                                      random_state=42)
    train_pool = Pool(train_X, train_y, cat_features=cat_features)
    eval_pool = Pool(val_X, val_y, cat_features=cat_features)

    if task == "train":
        # 训练分类模型，进行随机网格搜索
        params = {
            "loss_function": "MultiLogloss",
            # "auto_class_weights": "Balanced",  # 处理非平衡类别，其他选项有None，SqrtBalanced
            "eval_metric": "HammingLoss",
            "class_names": mlb.classes_,
            "od_type": "Iter",
            "cat_features": cat_features,
            "early_stopping_rounds": 100,
            "verbose": 100,
            "task_type": "CPU",
            "iterations": 250,
        }
        model = CatBoostClassifier(**params)
        grid = {
            "learning_rate": [0.03, 0.1],
            "depth": [4, 6, 10],
            "l2_leaf_reg": [1, 3, 5, 7, 9]
        }
        model.fit(train_pool)
        # model.randomized_search(grid,
        #                         train_pool,
        #                         verbose=50,
        #                         plot=True,
        #                         plot_file=str(pictures_path /
        #                                       "randomized_search.html"))
        pickle.dump(model, open(models_path / "model.pkl", "wb"))
    else:
        model = pickle.load(open(models_path / "model.pkl", "rb"))


if __name__ == "__main__":
    from datetime import datetime
    logger.add(f"./log/multi_label_classifier-{datetime.now().strftime('%Y-%m-%d')}.log",level="INFO")
    
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path",
                        type=str,
                        default="./cache/preprocessed_data_set.csv")
    parser.add_argument("--output_path", type=str, default="./results")
    parser.add_argument("--pictures_path", type=str, default="./pictures")
    parser.add_argument("--models_path", type=str, default="./models")
    parser.add_argument("--task", type=str, default="train")
    args = parser.parse_args()

    logger.info("Input Parameters informations:")
    message = "\n".join([f"{k:<20}: {v}" for k, v in vars(args).items()])
    logger.info(message)

    input_path = Path(args.input_path)
    output_path = Path(args.output_path)
    pictures_path = Path(args.pictures_path)
    models_path = Path(args.models_path)
    task = args.task
    for path in (output_path, pictures_path, models_path):
        if not path.exists():
            path.mkdir(parents=True)

    step(input_path, pictures_path, models_path, task)
