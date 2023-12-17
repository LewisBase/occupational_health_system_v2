# -*- coding: utf-8 -*-
"""
@DATE: 2023-12-13 10:45:51
@Author: Liu Hengjiang
@File: examples\time_series_predict\extract_data_for_tp-demo.py
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


def extract_diagnoise_res(x):
    from functional import seq
    set_res = seq(x.split(";")).map(lambda x: x.split("_")[1]).set()
    res = False if set_res.issubset({"目前未见异常","复查"}) else True
    return res

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
        "physical_exam_conclusion", "report_issue_date"
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
    init_df.dropna(subset=["report_issue_date","physical_exam_conclusion"], inplace=True)
    
    init_df["diagnoise_res"] = init_df[
        "physical_exam_conclusion"].parallel_apply(extract_diagnoise_res)
    init_df["hazard_num"] = init_df[
        "physical_exam_conclusion"].parallel_apply(extract_hazard_res)
    init_df["report_issue_date"] = pd.to_datetime(init_df["report_issue_date"])
    # init_df["year"] = pd.to_datetime(init_df["report_issue_date"]).dt.year
    # init_df["month"] = pd.to_datetime(init_df["report_issue_date"]).dt.month
    # init_df["year_month"] = init_df["year"].astype(str) + "-" + init_df["month"].astype(str)
    # groupby_df = init_df.groupby(["organization_city", "year_month"])["diagnoise_res"].sum().to_frame().reset_index()
    diagnoise_groupby_df = pd.DataFrame()
    diagnoise_groupby_df["exam_num"] = init_df.groupby(["organization_city", "report_issue_date"])["diagnoise_res"].size()
    diagnoise_groupby_df["diagnoise_num"] = init_df.groupby(["organization_city", "report_issue_date"])["diagnoise_res"].sum()
    diagnoise_groupby_df.reset_index(inplace=True)
    diagnoise_groupby_df.to_csv(output_path/"diagnoise_time_series_data.csv", header=True, index=False, encoding="utf-8-sig")
    
    hazard_groupby_df = init_df.groupby(["organization_city", "report_issue_date"])["hazard_num"].apply(lambda x: x.explode().value_counts()).to_frame().reset_index()
    hazard_groupby_df.rename(columns={"level_2":"hazard_type"}, inplace=True)
    hazard_groupby_df.to_csv(output_path/"hazard_time_series_data.csv", header=True, index=False, encoding="utf-8-sig")
    print(1)