# -*- coding: utf-8 -*-
"""
@DATE: 2023-12-12 15:50:19
@Author: Liu Hengjiang
@File: examples\single_label_classification\streamlit_show.py
@Software: vscode
@Description:
        职工上岗体检时，对其未来在岗天数中确诊职业病的概率进行预测并展示
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import shap
from loguru import logger
from pathlib import Path
from functional import seq
from catboost import Pool, CatBoostClassifier
from catboost.utils import get_roc_curve, get_fnr_curve, get_fpr_curve
from sklearn.model_selection import train_test_split
from sklearn import metrics

import sys

sys.path.append("../../../occupational_health_system_v2")

from utils.data_helper import get_categorical_indicies

if __name__ == "__main__":
    from matplotlib.font_manager import FontProperties
    from matplotlib import rcParams

    config = {
        "font.family": "serif",
        "font.size": 12,
        "mathtext.fontset":
        "stix",  # matplotlib渲染数学字体时使用的字体，和Times New Roman差别不大
        "font.serif": ["STZhongsong"],  # 华文中宋
        "axes.unicode_minus": False  # 处理负号，即-号
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

    input_df = pd.read_csv(input_path, header=0)
    labels = input_df["label"]
    features = input_df.drop(["id", "label"], axis=1)
    basic_feature_examples = [
        "age", "sex", "total_duration_month", "hazard_suffer_day", "duration"
    ]
    workshop_feature_examples = [
        "worktype",
        "physical_exam_hazard",
        "institution_location_code",
        "organization_name",
    ]
    health_feature_examples = [
        "Heart_result", "Liver_result", "Spleen_result", "Lung_result",
        "Skin_result", "ECG_result", "MPV_result", "PLCR_result", "MCV_result",
        "MCH_result"
    ]

    cat_features = get_categorical_indicies(features)
    features[cat_features] = features.fillna(value="", inplace=True)

    train_X, val_X, train_y, val_y = train_test_split(features,
                                                      labels,
                                                      train_size=0.8,
                                                      random_state=42)

    model = pickle.load(open(models_path / "model.pkl", "rb"))
    predict_y = model.predict(val_X)
    val_desc = seq(val_y).map(lambda x: "确诊" if x == 1 else "健康")
    predict_desc = seq(predict_y).map(lambda x: "确诊" if x == 1 else "健康")

    # 开始生成steamlit页面展示
    st.markdown("# 基于职工上岗体检结果的职业病患病预测模型")
    st.markdown("当一名职工进行完上岗体检时，可以根据其个人基本信息（年龄、性别、工龄等），\
        即将开始工作的岗位信息（雇主企业、工种、暴露危害因素等）以及本次的体检结果，\
        对未来在岗期间确诊职业病的概率进行预测。")
    st.markdown("## 数据示例")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("职工ID")
        st.dataframe(input_df["id"])
    with col2:
        st.markdown("职工基本信息")
        st.dataframe(input_df[basic_feature_examples])
    with col3:
        st.markdown("职工工作岗位信息")
        st.dataframe(input_df[workshop_feature_examples])
    with col4:
        st.markdown("职工职业健康信息")
        st.dataframe(input_df[health_feature_examples])

    st.markdown("## 批量预测结果展示")
    st.markdown(
        f"批量预测精确率: __{round(np.count_nonzero(val_y-predict_y==0)/len(val_y),2)}__"
    )
    st.dataframe(pd.DataFrame({"真实情况": val_desc, "预测结果": predict_desc}).T)

    st.markdown("## 单条职业健康记录预测")
    # 获取用户输入的数值
    default_value = val_X.iloc[-1]
    # 空间有限，仅展示少数
    for example_column in (basic_feature_examples, workshop_feature_examples,
                           health_feature_examples):
        cols = st.columns(len(example_column))
        user_values = default_value.copy()
        for i, col in enumerate(cols):
            with col:
                if input_df.columns.tolist().index(
                        example_column[i]) in cat_features:
                    input_list = input_df[
                        example_column[i]].value_counts().index.tolist()
                    try:
                        default_index = input_list.index( default_value[example_column].iloc[i])
                    except ValueError:
                        default_index = 0
                    value = st.selectbox(
                        label=example_column[i],
                        options=input_list,
                        index=default_index)
                else:
                    value = st.text_input(
                        f"{example_column[i]}",
                        value=default_value[example_column].iloc[i])
                user_values[example_column[i]] = value

    # 在点击按钮时进行运算
    if st.button("进行预测"):
        results = []
        proba_results = []
        for day in range(1, 366):
            user_values["duration"] = day
            results.append(model.predict(user_values.values))
            proba_results.append(model.predict_proba(user_values.values))

        fig, ax = plt.subplots(figsize=(10, 2), dpi=150)
        plot_results = seq(proba_results).map(lambda x: x[1]).list()
        p = ax.plot(range(1, 366), plot_results, marker="o", label="确诊概率")
        ax.set_xlabel("未来在岗天数")
        ax.set_ylabel("职业病确诊概率")
        plt.legend(loc="best")
        st.pyplot(fig)
        plt.close(fig)
