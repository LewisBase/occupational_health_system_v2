# -*- coding: utf-8 -*-
"""
@DATE: 2024-04-12 14:19:49
@Author: Liu Hengjiang
@File: examples\occhs_time_series_predict\extract_data_for_district_tp.py
@Software: vscode
@Description:
        提取数据进行各地区职业病确诊情况的时间序列预测
"""

import re
import ast
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from functional import seq
from loguru import logger
from pandarallel import pandarallel

from utils.database_helper import load_data_from_table, create_table_from_df


# def extract_diagnoise_res(dataframe):
#     diagnoise_dict = {
#         "上岗前职业健康检查": {"职业禁忌证"},
#         "在岗期间职业健康检查": {"职业禁忌证", "疑似职业病", "复查"},
#         "离岗时职业健康检查":{"疑似职业病", "复查", "职业禁忌证"},
#         "离岗后健康检查": {"疑似职业病", "复查", "职业禁忌证"},
#         "应急健康检查": {"疑似职业病", "复查", "职业禁忌证"}
#     }
#     from functional import seq
#     exam_conclusion = dataframe["physical_exam_conclusion"]
#     exam_type = dataframe["physical_exam_type"]
#     set_res = seq(exam_conclusion.str.split(";")).map(lambda x: seq(x).map(lambda y: y.split("_")[1]).set())
#     # 根据健康检查类型判断，健康检查结果与标准有交集即为True
#     res = seq(zip(exam_type, set_res)).map(lambda x : x[1] & diagnoise_dict.get(x[0])).list()
#     # res = False if set_res.issubset({"目前未见异常","其他疾病或异常"}) else True
#     return res
def extract_diagnoise_res(x):
    from functional import seq
    set_res = seq(x.split(";")).map(lambda x: x.split("_")[1]).set()
    return set_res


def extract_hazard_res(x):
    from functional import seq
    set_res = seq(x.split(";")).map(lambda x: x.split("_")[0]).set()
    return set_res


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--database_path",
        type=str,
        default=
        "D:\WorkingData\\2023浙江疾控数据\DataBase\occupational_health_2023.db")
    parser.add_argument("--input_table", type=str, default="BASIC_INFO")
    parser.add_argument("--output_path", type=str, default="./cache")
    args = parser.parse_args()

    logger.info("Input Parameters informations:")
    message = "\n".join([f"{k:<20}: {v}" for k, v in vars(args).items()])
    logger.info(message)

    database_path = Path(args.database_path)
    input_table = args.input_table
    output_path = Path(args.output_path)

    if not output_path.exists():
        output_path.mkdir(parents=True)

    pandarallel.initialize()

    columns_names = [
        "id", "report_card_id", "organization_city",
        "organization_industry_type", "organization_unified_social_credit_id",
        "physical_exam_type", "physical_exam_conclusion", "report_issue_date"
    ]
    filter_condition = {"organization_province": "浙江省"}
    limit_size = 0
    logger.info(f"Load data from database")
    init_df = load_data_from_table(database_path=database_path,
                                   table_name=input_table,
                                   column_names=columns_names,
                                   filter_condition=filter_condition,
                                   limit_size=limit_size)
    logger.info(f"Data Size: {init_df.shape}")
    #Data Size: (1712289, 5)

    # drop the line without date
    init_df.dropna(subset=["report_issue_date", "physical_exam_conclusion"],
                   inplace=True)

    # init_df["diagnoise_res"] = extract_diagnoise_res(init_df)
    init_df["diagnoise_res"] = init_df[
        "physical_exam_conclusion"].parallel_apply(extract_diagnoise_res)
    init_df["hazard_res"] = init_df["physical_exam_conclusion"].parallel_apply(
        extract_hazard_res)
    init_df["report_issue_date"] = pd.to_datetime(init_df["report_issue_date"])
    diagnoise_groupby_df = init_df.groupby(
        ["organization_city", "report_issue_date",
         "physical_exam_type"])["diagnoise_res"].apply(
             lambda x: x.explode().value_counts()).to_frame().reset_index()
    diagnoise_groupby_df.rename(columns={"level_3": "diagnoise_type"},
                                inplace=True)
    diagnoise_groupby_df.to_csv(output_path / "diagnoise_time_series_data.csv",
                                header=True,
                                index=False,
                                encoding="utf-8-sig")

    hazard_groupby_df = init_df.groupby([
        "organization_city", "report_issue_date", "organization_industry_type"
    ])["hazard_res"].apply(
        lambda x: x.explode().value_counts()).to_frame().reset_index()
    hazard_groupby_df.rename(columns={"level_3": "hazard_type"}, inplace=True)
    hazard_groupby_df.to_csv(output_path / "hazard_time_series_data.csv",
                             header=True,
                             index=False,
                             encoding="utf-8-sig")
    print(1)
