# -*- coding: utf-8 -*-
"""
@DATE: 2023-12-14 16:04:57
@Author: Liu Hengjiang
@File: examples\time_series_predict\streamlit_show.py
@Software: vscode
@Description:
        时间序列模型的streamlit页面展示
"""

import streamlit as st
import pandas as pd
import pickle
from loguru import logger
from pathlib import Path
from functional import seq

import sys

sys.path.append("../../../occupational_health_system_v2")
sys.path.append("/mount/src/occupational_health_system_v2") # 用于线上托管
from utils.data_helper import timeseries_train_test_split
from utils.plot_helper import plotly_forecast_res, plotly_forecast_trend, plotly_top_bar
from examples.occhs_time_series_predict.district_time_series_predict import CITY_NAMES

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
    
    diagnoise_input_df = pd.read_csv(input_path /
                                     "diagnoise_time_series_data.csv",
                                     header=0,
                                     nrows=nrows)
    diagnoise_input_df["report_issue_date"] = pd.to_datetime(
        diagnoise_input_df["report_issue_date"])
    diagnoise_input_df.sort_values(by="report_issue_date",
                                   ascending=True,
                                   inplace=True)
    hazard_input_df = pd.read_csv(input_path / "hazard_time_series_data.csv",
                                  header=0, nrows=nrows)
    hazard_input_df["report_issue_date"] = pd.to_datetime(
        hazard_input_df["report_issue_date"])
    hazard_input_df.sort_values(by="report_issue_date",
                                ascending=True,
                                inplace=True)

    return diagnoise_input_df, hazard_input_df


@st.cache_resource
def load_model(models_path, city_name):
    exam_model = pickle.load(
        open(models_path / f"{city_name}-diagnoise-exam_num-model.pkl", "rb"))
    diagnoise_model = pickle.load(
        open(models_path / f"{city_name}-diagnoise-diagnoise_num-model.pkl",
             "rb"))
    hazard_model = pickle.load(
        open(models_path / "top10_hazard_info.pkl", "rb"))
    city_name_chinese = seq(
        CITY_NAMES.items()).filter(lambda x: x[1] == city_name).list()
    hazard_res = hazard_model.get(city_name_chinese[0][0])
    return exam_model, diagnoise_model, hazard_res


def step(input_path, models_path):
    diagnoise_input_df, hazard_input_df = load_data(input_path, nrows=None)

    # 开始生成steamlit页面展示
    st.markdown("# 基于各地区职工职业病检测情况的时间序列模型")
    st.markdown("针对各地区体检机构上报的职工职业病体检情况进行汇总统计，按检测时间前后建立时间序列模型，对不同地区的职业病检测总人数、\
        职业病确诊人数以及高发职业病危害因素进行监控与预测。进一步得到职业病检测、确诊数据的变化趋势以及周期性变化规律。")

    st.markdown("## 数据示例")
    st.markdown(
        "选取所有记录中属于浙江省的检测记录，共计1712289条。按各地级市进行汇总，统计每个时间段进行职业病体检的人数、检测结果及职业危害暴露情况。"
    )
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("职工职业病体检情况")
        st.dataframe(diagnoise_input_df)
    with col2:
        st.markdown("职工职业危害因素暴露情况")
        st.dataframe(hazard_input_df)

    st.markdown("## 模型内容展示")
    st.markdown("可以观测到，在浙江省11个市中，最常出现的职业危害因素是噪声、粉尘与高温。温州、舟山、金华三地的苯类化学物质暴露也十分显著。")
    st.markdown("在职工进行职业健康体检的数据中，丽水、温州、舟山、衢州、金华等市进行职业健康体检的人数较少，杭州最多。")
    st.markdown("此外，丽水、温州、绍兴、宁波等市进行职业健康体检的人数近期呈下降趋势。这其中绍兴、宁波的确诊人数却呈上升趋势，需要重点关注。")
    cities = diagnoise_input_df["organization_city"].drop_duplicates().tolist()
    genre = st.radio("针对浙江省各地级市进行建模分析", cities, horizontal=True)
    logger.info(f"{genre} is selected.")
    city_name = CITY_NAMES.get(genre)

    exam_model, diagnoise_model, hazard_res = load_model(
        models_path=models_path, city_name=city_name)
    diagnoise_sub_df = diagnoise_input_df[
        diagnoise_input_df["organization_city"] == genre]
    train_X, test_X, train_y, test_y = timeseries_train_test_split(
        X=diagnoise_sub_df["report_issue_date"],
        y=diagnoise_sub_df[["exam_num", "diagnoise_num"]],
        train_size=0.8)

    with st.expander("职业危害因素暴露Top 10"):
        fig_res = plotly_top_bar(data=hazard_res)
        st.plotly_chart(fig_res, use_container_width=True)
    with st.expander("进行职业健康体检人数变化情况"):
        tab1, tab2 = st.tabs(["进行职业健康体检人数模型", "趋势分析"])
        periods = st.text_input("进行预测的时长（天）",
                                value=(test_X.iloc[-1] - test_X.iloc[0]).days +
                                1)
        future = exam_model.make_future_dataframe(periods=int(periods),
                                                  freq='D')
        forecast = exam_model.predict(future)
        with tab1:
            fig_res = plotly_forecast_res(model=exam_model,
                                        fcst=forecast,
                                        test_X=test_X,
                                        test_y=test_y["exam_num"],
                                        ylabel="exam_num",
                                        title=genre)
            st.plotly_chart(fig_res, use_container_width=True)
        with tab2:
            subtab1, subtab2, subtab3 = st.tabs(["整体趋势","月内趋势","周内趋势"])
            for subtab, titles in zip((subtab1, subtab2, subtab3),("trend", "monthly","weekly")):
                with subtab:
                    fig_res = plotly_forecast_trend(model=exam_model,
                                                  fcst=forecast,
                                                  ylabel=titles)
                    st.plotly_chart(fig_res, use_container_width=True)

    with st.expander("确诊罹患各类职业病人数变化情况"):
        tab1, tab2 = st.tabs(["确诊罹患各类职业病人数模型", "趋势分析"])
        periods = st.text_input("进行预测的时间（天）",
                                value=(test_X.iloc[-1] - test_X.iloc[0]).days +
                                1)
        future = diagnoise_model.make_future_dataframe(periods=int(periods),
                                                       freq='D')
        forecast = diagnoise_model.predict(future)
        with tab1:
            fig_res = plotly_forecast_res(model=diagnoise_model,
                                        fcst=forecast,
                                        test_X=test_X,
                                        test_y=test_y["diagnoise_num"],
                                        ylabel="diagnoise_num",
                                        title=genre)
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
    parser.add_argument("--input_path", type=str, default="../occhs_time_series_predict/cache")
    parser.add_argument("--models_path", type=str, default="../occhs_time_series_predict/models")
    # parser.add_argument("--input_path", type=str, default="/mount/src/occupational_health_system_v2/examples/occhs_time_series_predict/cache")
    # parser.add_argument("--models_path", type=str, default="/mount/src/occupational_health_system_v2/examples/occhs_time_series_predict/models")
    args = parser.parse_args()

    logger.info("Input Parameters informations:")
    message = "\n".join([f"{k:<20}: {v}" for k, v in vars(args).items()])
    logger.info(message)

    input_path = Path(args.input_path)
    models_path = Path(args.models_path)


    step(input_path, models_path)
