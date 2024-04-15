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
import plotly.graph_objects as go
import pickle
from loguru import logger
from pathlib import Path
from functional import seq
from catboost import Pool
from catboost.utils import get_roc_curve, get_fnr_curve, get_fpr_curve
from sklearn.model_selection import train_test_split
from sklearn import metrics

import sys

sys.path.append("../../../occupational_health_system_v2")
sys.path.append("/mount/src/occupational_health_system_v2") # 用于线上托管

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



@st.cache_data
def load_data(input_path, **kwargs):
    nrows = kwargs.pop("nrows", 200)
    
    input_df = pd.read_csv(input_path, header=0, nrows=nrows)
    labels = input_df["label"]
    features = input_df.drop(["id", "label"], axis=1)

    cat_features = get_categorical_indicies(features)
    features[cat_features] = features.fillna(value="", inplace=True)

    train_X, val_X, train_y, val_y = train_test_split(features,
                                                      labels,
                                                      train_size=0.8,
                                                      random_state=42)
    return input_df, cat_features, val_X, val_y


@st.cache_resource
def load_model_and_predict(models_path, val_X):
    model = pickle.load(open(models_path / "model.pkl", "rb"))
    predict_y = model.predict(val_X)
    return model, predict_y


def step(input_path, models_path):
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
    input_df, cat_features, val_X, val_y = load_data(input_path)
    eval_pool = Pool(val_X, val_y, cat_features=cat_features)
    model, predict_y = load_model_and_predict(models_path, val_X)

    # 开始生成steamlit页面展示
    st.markdown("# 基于职工岗前体检结果的职业病患病预测模型")
    st.markdown("当一名职工进行完上岗体检时，可以根据其个人基本信息（年龄、性别、工龄等），\
        即将开始工作的岗位信息（雇主企业、工种、暴露危害因素等）以及本次的体检结果，\
        对未来在岗期间确诊职业病的概率进行预测。")
    st.markdown("## 数据示例")
    st.markdown(
        "选取所有记录中职工在同一家公司供职且具有类型上岗、在岗或者上岗、离岗两个类别的体检记录数据作为建模的样本，共计14048条。")
    st.markdown("将上述样本对应的职工基本信息、工作岗位信息以及上岗体检中的所有检测结果作为特征进行训练。")
    st.markdown(
        "将样本中后一次体检记录中确诊为罹患职业病，或者有新增职业病类型的，标记为确诊；没有检测到职业病或者两次体检没有产生变化的，标记为健康。具体规则如下："
    )
    st.markdown("* 后一次体检结论中，所有结果均为前一次结果的子集，记为“未见异常”")
    st.markdown("* 后一次体检结论中，对各个环境暴露结果均为“目前未见异常”或者“复查”的，标记为“未见异常”")
    st.markdown("* 后一次体检结论中，与前一次体检结论之间存在差异且差异部分的结论为“目前未见异常”或者“复查”的，标记为“未见异常”")
    st.markdown("* 其于情况则标记为异常")
    st.markdown("最终结果中，标记为“异常”的样本共有4252条，标记为“未见异常”的样本共有9796条。")
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

    st.markdown("## 整体预测结果展示")
    st.markdown("将数据集分类两部分，80%用于训练模型，得到的模型在剩余20%的数据上进行预测。")
    st.markdown("模型的AUC达95%，精确率达77%，整体预测能力优秀。")
    st.markdown("继续整理规范并加工特征内容，可以进一步提升模型的预测能力。")
    st.markdown(
        f"整体预测精确率: __{round(np.count_nonzero(val_y-predict_y==0)/len(val_y),2)}__"
    )
    st.markdown("整体预测结果的性能指标：")
    # 绘制ROC曲线
    curve = get_roc_curve(model, eval_pool)
    (fpr, tpr, thresholds) = curve
    roc_auc = metrics.auc(fpr, tpr)
    lw = 2
    fig, axs = plt.subplots(1, 2, figsize=(10, 2))
    axs = np.ravel(axs)
    axs[0].plot(fpr,
                tpr,
                color='darkorange',
                lw=lw,
                label='ROC curve (area = %0.2f)' % roc_auc,
                alpha=0.5)
    axs[0].plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--', alpha=0.5)
    axs[0].set_xlabel("False Postive Rate")
    axs[0].set_ylabel("True Postive Rate")
    axs[0].legend(loc="best")
    axs[0].set_title("ROC Curve")
    # 绘制FPR、FNR曲线
    (thresholds, fpr) = get_fpr_curve(curve=curve)
    (thresholds, fnr) = get_fnr_curve(curve=curve)
    lw = 2
    axs[1].plot(thresholds, fpr, color='blue', lw=lw, label='FPR', alpha=0.5)
    axs[1].plot(thresholds, fnr, color='green', lw=lw, label='FNR', alpha=0.5)
    axs[1].set_xlabel("Threshold")
    axs[1].set_ylabel("Error Rate")
    axs[1].legend(loc="best")
    axs[1].set_title("FPR/FNR Curve")
    st.pyplot(fig)
    plt.close(fig)

    st.markdown("## 职工职业健康情况预测")
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
                        default_index = input_list.index(
                            default_value[example_column].iloc[i])
                    except ValueError:
                        default_index = 0
                    value = st.selectbox(label=example_column[i],
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
            user_values["duration"] = day # user_values["duration"] + day if user_values["duration"] is not None else day
            results.append(model.predict(user_values.values))
            proba_results.append(model.predict_proba(user_values.values))

        plot_results = seq(proba_results).map(lambda x: round(100*x[1],2)).list()
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=list(range(1, 366)), y=plot_results, mode="lines", name="体检结果异常概率"))
        fig.update_layout(xaxis_title="未来在岗天数",
                          yaxis_title="职业体检诊断异常概率(%)",
                          font=dict(family="Courier New, monospace",
                                    size=15,
                                    color="RebeccaPurple"))
        st.plotly_chart(fig, theme="streamlit", use_container_width=True)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    # parser.add_argument("--input_path",
    #                     type=str,
    #                     default="../occhs_single_label_classification/cache/preprocessed_data_set.csv")
    # parser.add_argument("--models_path", type=str, default="../occhs_single_label_classification/models")
    parser.add_argument("--input_path",
                        type=str,
                        default="/mount/src/occupational_health_system_v2/examples/occhs_single_label_classification/cache/preprocessed_data_set.csv")
    parser.add_argument("--models_path", type=str, default="/mount/src/occupational_health_system_v2/examples/occhs_single_label_classification/models")
    args = parser.parse_args()

    logger.info("Input Parameters informations:")
    message = "\n".join([f"{k:<20}: {v}" for k, v in vars(args).items()])
    logger.info(message)

    input_path = Path(args.input_path)
    models_path = Path(args.models_path)

    step(input_path, models_path)
