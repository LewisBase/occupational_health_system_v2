# -*- coding: utf-8 -*-
"""
@DATE: 2024-04-14 18:25:33
@Author: Liu Hengjiang
@File: examples\occupational_healthy_system-04_10\pages\\4_个体职工体检结果智能诊断.py
@Software: vscode
@Description:
        职工上岗体检时，对其多种不同类型职业病确诊概率进行预测并展示
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import pickle
from loguru import logger
from pathlib import Path
from functional import seq

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


import sys

sys.path.append("../../../occupational_health_system_v2")
sys.path.append("/mount/src/occupational_health_system_v2") # 用于线上托管

from utils.data_helper import get_categorical_indicies


@st.cache_data
def load_data(input_path, **kwargs):
    nrows = kwargs.pop("nrows", 200)
    
    input_df = pd.read_csv(input_path, header=0, nrows=nrows)
    labels = input_df["labels"]
    features = input_df.drop(["id", "labels"], axis=1)

    cat_features = get_categorical_indicies(features)
    features[cat_features] = features.fillna(value="", inplace=True)

    return input_df, cat_features, features, labels


@st.cache_resource
def load_model(models_path):
    model = pickle.load(open(models_path / "model.pkl", "rb"))
    return model


def step(input_path, models_path):
    basic_feature_examples = [
        "age", "sex", "total_duration_month", "hazard_suffer_day"
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
    input_df, cat_features, val_X, val_y = load_data(input_path, nrows=200)
    model = load_model(models_path)

    # 开始生成steamlit页面展示
    st.markdown("# 基于职工体检结果预测可能患有的职业病类型的智能诊断模型")
    st.markdown("输入单个职工的体检结果报告，通过模型分析后，输出职工患有不同类型职业病的概率。")
    st.markdown("## 数据示例")
    st.markdown(
        "选取所有记录中职工在同一家公司供职且具有类型上岗、在岗或者上岗、离岗两个类别的体检记录数据作为建模的样本。")
    st.markdown("将上述样本对应的职工基本信息、工作岗位信息以及上岗体检中的所有检测结果作为特征进行训练。")
    st.markdown("根据职工体检报告中的体检结论与体检结论详情对用户确诊的职业病类型进行分类，具体类别与之前全省职业病累计确诊模型中的一致。")
    st.markdown("由于个别职业病类型在选取的数据集中出现次数较少，这里仅针对出现数量较多的职业病类型进行学习。")
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

    st.markdown("## 职工体检结果智能诊断")
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
        diseases_name = ["其他疾病或异常", "疑似职业病", "目前未见异常", "职业性中毒性肝病", "职业性中毒性肾病", "职业性内分泌系统疾病",
       "职业性听力损伤", "职业性呼吸系统疾病", "职业性心血管系统系统疾病", "职业性泌尿生殖系统疾病", "职业性眼病"]
        proba_results = seq(model.predict_proba(user_values.values)).map(lambda x: round(100 * x,2)).list()

        logger.info(f"{proba_results}")
        fig = go.Figure()
        fig.add_trace(go.Bar(x=diseases_name, y=proba_results, name="不同类型职业病确诊概率"))
        fig.update_layout(xaxis_title="智能诊断结果",
                          yaxis_title="职业病确诊概率(%)",
                          font=dict(family="Courier New, monospace",
                                    size=15,
                                    color="RebeccaPurple"))
        st.plotly_chart(fig, theme="streamlit", use_container_width=True)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    # parser.add_argument("--input_path",
    #                     type=str,
    #                     default="/mount/src/occupational_health_system_v2/examples/occhs_multi_label_classification/cache/preprocessed_data_set.csv")
    # parser.add_argument("--models_path", type=str, default="/mount/src/occupational_health_system_v2/examples/occhs_multi_label_classification/models")
    parser.add_argument("--input_path",
                        type=str,
                        default="../occhs_multi_label_classification/cache/preprocessed_data_set.csv")
    parser.add_argument("--models_path", type=str, default="../occhs_multi_label_classification/models")
    args = parser.parse_args()

    logger.info("Input Parameters informations:")
    message = "\n".join([f"{k:<20}: {v}" for k, v in vars(args).items()])
    logger.info(message)

    input_path = Path(args.input_path)
    models_path = Path(args.models_path)

    step(input_path, models_path)
