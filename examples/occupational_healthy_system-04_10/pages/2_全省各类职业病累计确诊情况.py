# -*- coding: utf-8 -*-
"""
@DATE: 2024-04-11 21:23:10
@Author: Liu Hengjiang
@File: examples\occupational_healthy_system-04_10\pages\\2_全省各类职业病累计确诊情况.py
@Software: vscode
@Description:
        全省各类职业病累计确诊情况
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import folium
import pickle
from streamlit_folium import folium_static
from loguru import logger
from pathlib import Path
from functional import seq
from prophet import Prophet

import sys

sys.path.append("../../../occupational_health_system_v2")
sys.path.append("/mount/src/occupational_health_system_v2") # 用于线上托管
from utils.data_helper import timeseries_train_test_split
from utils.plot_helper import plotly_forecast_res, plotly_forecast_trend, plotly_top_bar
from examples.time_series_predict.disease_time_series_predict import OCCUPATIONAL_DISEASE_TYPE_NAME

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
def load_data(input_path):
    diagnoise_input_df = pd.read_csv(input_path /
                                     "disease_diagnoise_time_series_data.csv",
                                     header=0)
    diagnoise_input_df["report_issue_date"] = pd.to_datetime(
        diagnoise_input_df["report_issue_date"])
    diagnoise_input_df.sort_values(by="report_issue_date",
                                   ascending=True,
                                   inplace=True)
    return diagnoise_input_df


@st.cache_resource
def load_model(models_path, disease):
    diagnoise_model = pickle.load(
        open(models_path / f"{disease}-diagnoise-model.pkl",
             "rb"))
    disease_name_chinese = seq(
        OCCUPATIONAL_DISEASE_TYPE_NAME.items()).filter(lambda x: x[1] == disease).list()
    return diagnoise_model


def step(input_path, models_path):
    diagnoise_input_df = load_data(input_path)

    # 开始生成steamlit页面展示
    st.markdown("# 基于各类职业病累计确诊情况的时间序列模型")
    st.markdown("针对各地区体检机构上报的职工职业病确诊结果进行汇总统计，按检测时间前后建立时间序列模型，\
        对不同类型职业病的累计确诊总人数进行监控与预测。得到不同职业病类型确诊数据的变化趋势以及周期性变化规律。")

    st.markdown("## 数据示例")
    st.markdown(
        "选取所有记录中属于浙江省的检测记录，共计1712289条。按体检报告中医生给出的描述对确诊的职业病进行简单分类，\
            统计每个时间段每类职业病的累计确诊病例数进行建模分析。"
    )
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("职业病类别")
        st.markdown("""
                    """)
    with col2:
        st.markdown("各类职业病累计确诊情况")
        st.dataframe(diagnoise_input_df)

    st.markdown("## 模型内容展示")
    st.markdown("可以观测到，在浙江省2021-2022年度的体检数据中，职业性听力损伤是占比最突出的主要职业病类型，其次为心血管疾病与呼吸系统疾病。")
    
    with st.expander("主要职业病类型累计确诊人数整体比例分布"):
        disease_sum = diagnoise_input_df.groupby(["diagnoise_res"]).sum()
        disease_sum.drop(["其他疾病或异常"],inplace=True)
        disease_sum_dict = (disease_sum/disease_sum.sum()).to_dict()["diagnoise_num"]
        fig_res = plotly_top_bar(data=disease_sum_dict)
        st.plotly_chart(fig_res, use_container_width=True)

    disease_list = diagnoise_input_df["diagnoise_res"].drop_duplicates().tolist()
    disease_list.remove("职业性肿瘤")
    disease_list.remove("其他疾病或异常")
    st.markdown("在具体的累计确诊趋势中，个别类型的职业病由于确诊的记录信息较少，目前还难以构建可靠的趋势预测模型。")
    genre = st.radio("针对浙江省各类职业病累计确诊情况进行建模分析", disease_list, horizontal=True)
    logger.info(f"{genre} is selected.")
    disease_name = OCCUPATIONAL_DISEASE_TYPE_NAME.get(genre)
    diagnoise_sub_df = diagnoise_input_df[
        diagnoise_input_df["diagnoise_res"] == genre].copy()
    diagnoise_sub_df["diagnoise_cumsum"] = diagnoise_sub_df["diagnoise_num"].cumsum()
    train_X, test_X, train_y, test_y = timeseries_train_test_split(
        X=diagnoise_sub_df["report_issue_date"],
        y=diagnoise_sub_df["diagnoise_cumsum"],
        train_size=0.8)

    with st.expander("各类职业病累计全确诊人数变化情况"):
        tab1, tab2 = st.tabs(["各类职业病累计确诊人数模型", "趋势分析"])
        periods = st.text_input("进行预测的时间（天）",
                                value=(test_X.iloc[-1] - test_X.iloc[0]).days + 22)
        diagnoise_model = load_model(
            models_path=models_path, disease=disease_name)
        future = diagnoise_model.make_future_dataframe(periods=int(periods),
                                                       freq='D')
        forecast = diagnoise_model.predict(future)
        with tab1:
            fig_res = plotly_forecast_res(model=diagnoise_model,
                                        fcst=forecast,
                                        test_X=test_X,
                                        test_y=test_y,
                                        ylabel="diagnoise_cumsum",
                                        title=genre,
                                        legend_xy=(0.1, 1.0))
            st.plotly_chart(fig_res, use_container_width=True)
        with tab2:
            subtab1, subtab2, subtab3 = st.tabs(["整体趋势","月内趋势","周内趋势"])
            for subtab, titles in zip((subtab1, subtab2, subtab3),("trend", "monthly","weekly")):
                with subtab:
                    fig_res = plotly_forecast_trend(model=diagnoise_model,
                                                  fcst=forecast,
                                                  ylabel=titles)
                    st.plotly_chart(fig_res, use_container_width=True)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    # parser.add_argument("--input_path", type=str, default="../time_series_predict/cache")
    # parser.add_argument("--models_path", type=str, default="../time_series_predict/models")
    parser.add_argument("--input_path", type=str, default="/mount/src/occupational_health_system_v2/examples/time_series_predict/cache")
    parser.add_argument("--models_path", type=str, default="/mount/src/occupational_health_system_v2/examples/time_series_predict/models")
    args = parser.parse_args()

    logger.info("Input Parameters informations:")
    message = "\n".join([f"{k:<20}: {v}" for k, v in vars(args).items()])
    logger.info(message)

    input_path = Path(args.input_path)
    models_path = Path(args.models_path)


    step(input_path, models_path)
