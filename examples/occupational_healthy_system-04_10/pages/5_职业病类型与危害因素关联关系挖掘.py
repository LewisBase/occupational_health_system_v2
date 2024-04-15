# -*- coding: utf-8 -*-
"""
@DATE: 2024-04-15 15:13:28
@Author: Liu Hengjiang
@File: examples\occhs_association_rule_mining\streamlit_show.py
@Software: vscode
@Description:
        环境危害因素暴露与职业病关系挖掘结果展示
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import pickle
from loguru import logger
from pathlib import Path
from functional import seq
from catboost import Pool

import sys

sys.path.append("../../../occupational_health_system_v2")
sys.path.append("/mount/src/occupational_health_system_v2") # 用于线上托管

from examples.occhs_time_series_predict.disease_time_series_predict import OCCUPATIONAL_DISEASE_TYPE_NAME


@st.cache_data
def load_data(input_path, **kwargs):
    nrows = kwargs.pop("nrow", 200)
    input_df = pd.read_csv(input_path, header=0, nrows=nrows)
    return input_df


@st.cache_resource
def load_model(models_path, disease_name):
    pkl_path = models_path / f"{OCCUPATIONAL_DISEASE_TYPE_NAME.get(disease_name)}_data_mining_model.pkl"
    frq_items, rules = pickle.load(open(pkl_path, "rb"))
    return frq_items, rules


def step(input_path, models_path):
    input_df = load_data(input_path)

    # 开始生成steamlit页面展示
    st.markdown("# 基于职工体检结果对危害因素与职业病类型之间的关联关系挖掘")
    st.markdown("统计所有职工的体检结果中所暴露的环境危害类型与确诊的职业病类型，进行二者之间的关联关系挖掘，得到诱发不同类型职业病最关键的危害因素。")
    st.markdown("## 数据示例")
    st.markdown("选取所有记录中职工确诊的职业病类型与其所暴露的危害因素类型。")
    st.markdown("将上述样本按不同职业病类型归类后依次进行关联关系挖掘。")
    st.markdown("由于与具体工厂环境数据进行匹配耗时过长，此处职工所暴露的危害因素来自于其体检报告中的记录，尚未涉及具体的暴露程度。")
    st.dataframe(input_df)

    st.markdown("## 关联关系挖掘结果")
    disease_list = seq(OCCUPATIONAL_DISEASE_TYPE_NAME.keys()).filter(lambda x: x not in ("疑似职业病", "其他疾病或异常"))
    genre = st.radio("针对各类职业病最相关的危害因素进行挖掘分析", disease_list, horizontal=True)
    logger.info(f"{genre} is selected.")
    frq_items, rules = load_model(models_path, genre)
    hazard_type_list = seq(frq_items["itemsets"].values).map(lambda x: x.split("\'")[1]).list()
    # 创建节点列表和边列表
    nodes = [genre] + hazard_type_list
    edges = [(genre, hazard_type) for hazard_type in hazard_type_list]
    # 创建节点坐标
    node_positions = {genre: [0, 0]}
    n = len(hazard_type_list) // 2 * -1
    try:
        point_gap = (len(hazard_type_list) // 2 * 2) / (len(hazard_type_list) - 1)
    except ZeroDivisionError:
        point_gap = 0
    for hazard_type in hazard_type_list:
        node_positions.update({hazard_type: [1, n]})
        n += point_gap
    # 创建节点的 X 和 Y 坐标列表
    x_nodes = [node_positions[node][0] for node in nodes]
    y_nodes = [node_positions[node][1] for node in nodes]

    # 创建边的起始点和终止点坐标列表
    x_edges = []
    y_edges = []
    for edge in edges:
        x_edges += [node_positions[edge[0]][0], node_positions[edge[1]][0], None]
        y_edges += [node_positions[edge[0]][1], node_positions[edge[1]][1], None]

    # 创建节点的散点图
    node_trace = go.Scatter(
        x=x_nodes,
        y=y_nodes,
        mode="markers+text",
        marker=dict(symbol="circle", size=50, color="blue"),
        text=nodes,
        textposition="bottom center",
        hoverinfo="text"
    )

    # 创建边的散点图
    edge_trace = go.Scatter(
        x=x_edges,
        y=y_edges,
        mode="lines",
        line=dict(width=1, color="gray"),
        hoverinfo="none"
    )

    # 创建图形数据
    data = [node_trace, edge_trace]

    # 创建图形布局
    layout = go.Layout(
        title="关联关系图",
        showlegend=False,
        hovermode="closest",
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
    )

    # 创建图形对象
    fig_res = go.Figure(data=data, layout=layout)
    st.plotly_chart(fig_res, use_container_width=False)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path",
                        type=str,
                        default="/mount/src/occupational_health_system_v2/examples/occhs_association_rule_mining/cache/disease_hazard_group_data.csv")
    parser.add_argument("--models_path", type=str, default="/mount/src/occupational_health_system_v2/examples/occhs_association_rule_mining/models")
    # parser.add_argument("--input_path",
    #                     type=str,
    #                     default="../occhs_association_rule_mining/cache/disease_hazard_group_data.csv")
    # parser.add_argument("--models_path", type=str, default="../occhs_association_rule_mining/models")
    args = parser.parse_args()

    logger.info("Input Parameters informations:")
    message = "\n".join([f"{k:<20}: {v}" for k, v in vars(args).items()])
    logger.info(message)

    input_path = Path(args.input_path)
    models_path = Path(args.models_path)

    step(input_path, models_path)
