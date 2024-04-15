# -*- coding: utf-8 -*-
"""
@DATE: 2024-04-12 10:54:17
@Author: Liu Hengjiang
@File: examples\occupational_healthy_system-04_10\pages\\2_全省各类职业病累计确诊情况.py
@Software: vscode
@Description:
        全省各类职业病累计确诊情况
"""

import streamlit as st
import pandas as pd
import pickle
from loguru import logger
from pathlib import Path
from functional import seq

import sys

sys.path.append("../../../occupational_health_system_v2")
sys.path.append("/mount/src/occupational_health_system_v2")  # 用于线上托管
from utils.data_helper import timeseries_train_test_split
from utils.plot_helper import plotly_forecast_res, plotly_forecast_trend, plotly_top_bar
from examples.occhs_time_series_predict.disease_time_series_predict import OCCUPATIONAL_DISEASE_TYPE_NAME

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


@st.cache_data
def load_data(input_path, **kwargs):
    nrows = kwargs.pop("nrows", 200)

    diagnoise_input_df = pd.read_csv(input_path /
                                     "disease_diagnoise_time_series_data.csv",
                                     header=0,
                                     nrows=nrows)
    diagnoise_input_df["report_issue_date"] = pd.to_datetime(
        diagnoise_input_df["report_issue_date"])
    diagnoise_input_df.sort_values(by="report_issue_date",
                                   ascending=True,
                                   inplace=True)
    disease_sum = diagnoise_input_df.groupby(["diagnoise_res"]).sum()
    disease_sum.drop(["其他疾病或异常"], inplace=True)
    disease_sum_dict = (disease_sum /
                        disease_sum.sum()).to_dict()["diagnoise_num"]
    return diagnoise_input_df, disease_sum_dict


@st.cache_resource
def load_model(models_path, disease):
    diagnoise_model = pickle.load(
        open(models_path / f"{disease}-diagnoise-model.pkl", "rb"))
    return diagnoise_model


def step(input_path, models_path):
    diagnoise_input_df, disease_sum_dict = load_data(input_path, nrows=None)

    # 开始生成steamlit页面展示
    st.markdown("# 基于各类职业病累计确诊情况的时间序列模型")
    st.markdown("针对各地区体检机构上报的职工职业病确诊结果进行汇总统计，按检测时间前后建立时间序列模型，\
        对不同类型职业病的累计确诊总人数进行监控与预测。得到不同职业病类型确诊数据的变化趋势以及周期性变化规律。")

    st.markdown("## 数据示例")
    st.markdown("选取所有记录中属于浙江省的检测记录，共计1712289条。按体检报告中医生给出的描述对确诊的职业病进行简单分类，\
            统计每个时间段每类职业病的累计确诊病例数进行建模分析。")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### 职业病类别")
        st.markdown("""
                    目前的体检报告信息中给出了体检结论与体检结论详情两部分关键内容，
                    体检结论中包含职工所暴露的职业健康危害因素及其像对应的体检结果，
                    具体的体检结果有以下几类：
                    * 目前未见异常；
                    * 复查；
                    * 其他疾病或异常；
                    * 疑似职业病；
                    * 职业禁忌症；

                    这里保留体检结论中的“目前未见异常”、“其他疾病或异常”、“疑似职业病”作为体检结果标签，
                    对于“职业禁忌证”的情况，进一步根据体检结果详情中的描述内容细分为以下13种明细类别：

                    |职业禁忌症类型|职业禁忌症类型|
                    |:----|:----|
                    |职业性听力损伤          |职业性眼病|
                    |职业性皮肤病            |职业性中毒性肾病|
                    |职业性心血管系统系统疾病 |职业性中毒性肝病|
                    |职业性呼吸系统疾病       |职业性肿瘤|
                    |职业性内分泌系统疾病     |职业性放射性疾病|
                    |职业性泌尿生殖系统疾病    |职业性骨关节疾病|
                    |职业性神经系统疾病        ||

                    > 目前针对职业禁忌症的归类主要根据文字描述中出现的关键字进行分类，后续可结合专业医师的鉴定进行更加准确的分类。
                    """)
    with col2:
        st.markdown("### 各类职业病累计确诊数据示例")
        st.dataframe(diagnoise_input_df)

    st.markdown("## 模型内容展示")
    st.markdown(
        "可以观测到，在浙江省2021-2022年度的体检数据中，职业性听力损伤是占比最突出的主要职业病类型，其次为心血管疾病与呼吸系统疾病。")

    with st.expander("主要职业病类型累计确诊人数整体比例分布"):
        fig_res = plotly_top_bar(data=disease_sum_dict)
        st.plotly_chart(fig_res, use_container_width=True)

    disease_list = seq(OCCUPATIONAL_DISEASE_TYPE_NAME.keys()).filter(
        lambda x: x not in ("职业性肿瘤", "其他疾病或异常"))
    st.markdown("""
                在具体的累计确诊趋势中，个别类型的职业病由于确诊的记录信息较少，目前还难以构建可靠的趋势预测模型。
                
                此外还可以看到，对于听力损伤、呼吸系统疾病以及心血管系统疾病几类主要的职业禁忌症类别，几乎每天都有新增确诊的案例，相对其他几类来说确诊形势更加严峻。
                """)
    genre = st.radio("针对浙江省各类职业病累计确诊情况进行建模分析", disease_list, horizontal=True)
    logger.info(f"{genre} is selected.")
    disease_name = OCCUPATIONAL_DISEASE_TYPE_NAME.get(genre)
    diagnoise_sub_df = diagnoise_input_df[diagnoise_input_df["diagnoise_res"]
                                          == genre].copy()
    diagnoise_sub_df["diagnoise_cumsum"] = diagnoise_sub_df[
        "diagnoise_num"].cumsum()
    train_X, test_X, train_y, test_y = timeseries_train_test_split(
        X=diagnoise_sub_df["report_issue_date"],
        y=diagnoise_sub_df["diagnoise_cumsum"],
        train_size=0.8)

    with st.expander("各类职业病累计全确诊人数变化情况"):
        tab1, tab2 = st.tabs(["各类职业病累计确诊人数模型", "趋势分析"])
        periods = st.text_input("进行预测的时间（天）",
                                value=(test_X.iloc[-1] - test_X.iloc[0]).days +
                                22)
        diagnoise_model = load_model(models_path=models_path,
                                     disease=disease_name)
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
            subtab1, subtab2, subtab3 = st.tabs(["整体趋势", "月内趋势", "周内趋势"])
            for subtab, titles in zip((subtab1, subtab2, subtab3),
                                      ("trend", "monthly", "weekly")):
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
    # parser.add_argument(
    #     "--input_path",
    #     type=str,
    #     default=
    #     "/mount/src/occupational_health_system_v2/examples/occhs_time_series_predict/cache"
    # )
    # parser.add_argument(
    #     "--models_path",
    #     type=str,
    #     default=
    #     "/mount/src/occupational_health_system_v2/examples/occhs_time_series_predict/models"
    # )
    args = parser.parse_args()

    logger.info("Input Parameters informations:")
    message = "\n".join([f"{k:<20}: {v}" for k, v in vars(args).items()])
    logger.info(message)

    input_path = Path(args.input_path)
    models_path = Path(args.models_path)

    step(input_path, models_path)
