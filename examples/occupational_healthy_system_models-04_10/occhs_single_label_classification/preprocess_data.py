# -*- coding: utf-8 -*-
"""
@DATE: 2023-12-08 09:38:02
@Author: Liu Hengjiang
@File: examples\initial_test\preprocess_data.py
@Software: vscode
@Description:
        对DEMO中要用到的数据进行特征工程
"""

import re
import ast
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from scipy import stats
from pathlib import Path
from functional import seq
from loguru import logger
from joblib import Parallel, delayed

from utils.database_helper import load_data_from_table
from utils.data_helper import reduce_mem_usage, drop_duplicate_columns, drop_unique_value_columns, array_padding

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--database_path",
        type=str,
        default=
        "D:\WorkingData\\2023浙江疾控数据\DataBase\occupational_health_2023.db")
    parser.add_argument("--feature_label_table",
                        type=str,
                        default="TOTAL_FEATURE_AND_LABEL_DEMO")
    parser.add_argument("--output_path", type=str, default="./cache")
    args = parser.parse_args()

    logger.info("Input Parameters informations:")
    message = "\n".join([f"{k:<20}: {v}" for k, v in vars(args).items()])
    logger.info(message)

    database_path = args.database_path
    feature_label_table = args.feature_label_table
    output_path = args.output_path

    original_df = load_data_from_table(database_path=database_path,
                                       table_name=feature_label_table,
                                       column_names=[],
                                       filter_condition={},
                                       limit_size=0)

    # 特殊列进行数据处理
    preprocessed_df = original_df.copy()
    # type convert
    logger.info(f"Convert label and birthday")
    preprocessed_df["label"] = preprocessed_df["label"].astype(int)
    preprocessed_df["birthday"] = pd.to_datetime(preprocessed_df["birthday"])
    # calculate age
    logger.info(f"Calculate age")
    current_date = datetime.now()
    preprocessed_df["age"] = (current_date - preprocessed_df["birthday"]
                              ) // pd.Timedelta(days=365.25)
    ## 对于出现负值的情况，进行取绝对值（可能是两项日期的前后位置填写错误导致）
    preprocessed_df["hazard_suffer_day"] = np.abs(
        (pd.to_datetime(preprocessed_df["physical_exam_date"]) -
         pd.to_datetime(preprocessed_df["hazard_start_date"])).dt.days)
    ## 暴露时长大于年龄的，置为NAN
    preprocessed_df.loc[preprocessed_df[
        preprocessed_df["hazard_suffer_day"] > preprocessed_df["age"] *
        365].index, "hazard_suffer_day"] = np.nan
    # calculate total duration month
    logger.info(f"Calculate total duration month")
    preprocessed_df["total_duration_month"] = preprocessed_df[
        "duration_year"].apply(lambda x: 0 if x is None else int(x[:-1]) * 12
                               ) + preprocessed_df["duration_month"].apply(
                                   lambda x: 0 if x is None else int(x[:-1]))
    ## 工龄大于年龄的，置为NAN
    preprocessed_df.loc[preprocessed_df[
        preprocessed_df["total_duration_month"] > preprocessed_df["age"] *
        12].index, "total_duration_month"] = np.nan

    # expand hazard_code
    preprocessed_df["hazard_code_num"] = preprocessed_df[
        "hazard_code"].str.split(",").apply(lambda x: len(x))
    hazard_code_expand = array_padding(
        origin_data=preprocessed_df["hazard_code"].str.split(","),
        constant_values=np.nan)
    hazard_code_expand_df = pd.DataFrame(
        hazard_code_expand,
        columns=[
            "hazard_code_" + str(i) for i in range(hazard_code_expand.shape[1])
        ])
    preprocessed_df = preprocessed_df.join(hazard_code_expand_df.iloc[:, :10],
                                           how="inner")

    # drop columns
    logger.info(f"Drop useless columns")
    drop_columns = [
        "report_institution", "organization_province", "organization_city",
        "organization_county", "employing_unit_province",
        "employing_unit_city", "employing_unit_county", "physical_exam_type",
        "is_reexam", "physical_exam_conclusion",
        "physical_exam_conclusion_detail", "report_issue_date", "report_date",
        "physical_exam_id", "report_card_id", "birthday", "duration_year",
        "duration_month", "hazard", "hazard_start_date", "physical_exam_date",
        "hazard_code"
    ]
    preprocessed_df.drop(drop_columns, axis=1, inplace=True)

    # drop content duplicate columns
    logger.info(f"Drop columns with same content")
    preprocessed_df = drop_duplicate_columns(preprocessed_df)

    # drop columns with all same value
    logger.info(f"Drop columns with all same value")
    preprocessed_df = drop_unique_value_columns(preprocessed_df)

    # 最终格式转换
    hazard_code_features = seq(
        preprocessed_df.columns).filter(lambda x: x.startswith("hazard_code_")
                                        and not x.endswith("num")).list()
    preprocessed_df[hazard_code_features] = preprocessed_df[
        hazard_code_features].astype(str)
    other_code_features = seq(
        preprocessed_df.columns).filter(lambda x: x.endswith("_code")).list()
    preprocessed_df[other_code_features] = preprocessed_df[
        other_code_features].astype(str)

    # 来自于pandas加载文件时的自动识别
    special_cols = [122, 137, 155, 158, 197, 230, 299, 563, 566]
    pattern_dict = {
        r"(R 1\.0   L 1\.0)|(&lt;1\.00)|(&lt;1\.0)": "1.0",
        r"(5\.0矫正)|(矫正5\.0)": "5.0",
        r"&gt;26\.5": "26.5",
        r"2\.0   g/L": "2.0",
        r"0\.5   g/L": "0.5",
        r"3\.0   g/L": "3.0",
        r"(-)|(正常)|(未检)|(有光感)|(未见异常)|(/)|(未见明显异常)|(未查)|(无)": "0.0",
    }
    for col in special_cols:
        for pattern, num in pattern_dict.items():
            preprocessed_df.iloc[:, col] = preprocessed_df.iloc[:, col].apply(
                lambda x: re.sub(pattern, num, x) if isinstance(x, str) else x)
        preprocessed_df.iloc[:, col] = preprocessed_df.iloc[:, col].astype(float)

    # 压缩内存
    logger.info(f"Compress memory usage")
    preprocessed_df = reduce_mem_usage(preprocessed_df)

    output_file = Path(output_path) / "preprocessed_data_set.csv"
    preprocessed_df.to_csv(output_file,
                           header=True,
                           index=False,
                           encoding="utf-8-sig")

    print(1)
