# -*- coding: utf-8 -*-
"""
@DATE: 2023-11-29 09:29:15
@Author: Liu Hengjiang
@File: examples\initial_test\extract_data_for_ml-demo.py
@Software: vscode
@Description:
        利用python+sqlite进行机器学习任务的数据筛选工作
        拟选取同时具有上岗前、离岗时数据的样本作为学习对象，探究在岗过程中可能导致职业病发生的情况

        进行样本打标，特征提取
"""

import re
import ast
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sqlite3 as sl
from pathlib import Path
from functional import seq
from loguru import logger
from joblib import Parallel, delayed

from utils.database_helper import load_data_from_table, create_table_from_df


def isnot_dict_subset(row: pd.DateOffset, parent_col: str,
                      sub_col: str) -> bool:
    parent_dict = row[parent_col]
    sub_dict = row[sub_col]
    return not set(sub_dict.items()).issubset(set(parent_dict.items()))


def is_dict_diff_subset(row: pd.DataFrame, before_col: str, after_col: str,
                        parent_set: set) -> set:
    before_dict = row[before_col]
    after_dict = row[after_col]
    diff = dict(set(after_dict.items()) - set(before_dict.items()))
    res = set(
        [value for key, value in diff.items() if key in after_dict.keys()])
    return res.issubset(parent_set)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--database_path",
        type=str,
        default=
        "D:\WorkingData\\2023浙江疾控数据\DataBase\occupational_health_2023.db")
    parser.add_argument("--input_table", type=str, default="BASIC_INFO_DEMO")
    parser.add_argument("--labeled_table",
                        type=str,
                        default="BASIC_INFO_AND_LABEL_DEMO")
    parser.add_argument("--feature_tables",
                        type=list,
                        default=[
                            "PHYSI_EXAMI_I", "PHYSI_EXAMI_II",
                            "PHYSI_EXAMI_III", "PHYSI_EXAMI_IV",
                            "PHYSI_EXAMI_V"
                        ])
    parser.add_argument("--total_feature_label_table",
                        type=str,
                        default="TOTAL_FEATURE_AND_LABEL_DEMO")

    # parser.add_argument("--task", type=str, default="construct_label")
    parser.add_argument("--task", type=str, default="extract_feature")

    args = parser.parse_args()

    logger.info("Input Parameters informations:")
    message = "\n".join([f"{k:<20}: {v}" for k, v in vars(args).items()])
    logger.info(message)

    database_path = args.database_path
    input_table = args.input_table
    labeled_table = args.labeled_table
    feature_tables = args.feature_tables
    total_feature_label_table = args.total_feature_label_table

    task = args.task

    if task == "construct_label":
        column_names = [
            "report_card_id", "physical_exam_id", "institution_code",
            "report_institution", "institution_location_code",
            "organization_province", "organization_city",
            "organization_county", "organization_location_code",
            "organization_name", "organization_enterprise_type",
            "organization_industry_type", "organization_industry_type_code",
            "organization_enterprise_scale", "employing_unit",
            "employing_unit_enterprise_type", "employing_unit_industry_type",
            "employing_unit_enterprise_scale", "employing_unit_province",
            "employing_unit_city", "employing_unit_county",
            "employing_unit_location_code", "sex", "birthday",
            "physical_exam_type", "physical_exam_date", "worktype_code",
            "worktype", "worktype_other", "hazard_code", "hazard",
            "hazard_start_date", "physical_exam_hazard", "duration_year",
            "duration_month", "is_reexam", "detection_type",
            "physical_exam_conclusion", "physical_exam_conclusion_detail",
            "report_issue_date", "report_date", "id"
        ]
        filter_condition = {}
        limit_size = 0
        logger.info("Load data from database")
        init_df = load_data_from_table(database_path=database_path,
                                       table_name=input_table,
                                       column_names=column_names,
                                       filter_condition=filter_condition,
                                       limit_size=limit_size)

        # Step 1: 首先按照体检类型进行筛选
        # 选出同时具有两条体检记录，且类别分别为上岗、在岗或者上岗、离岗的数据
        logger.info("Filter useful data")
        useful_exam_types = ["上岗前职业健康检查", "在岗期间职业健康检查", "离岗时职业健康检查"]
        exam_type_total_df = init_df[init_df["physical_exam_type"].isin(
            useful_exam_types)]
        filter_id_list = exam_type_total_df.groupby("id").filter(
            lambda x: (set(x["physical_exam_type"]) == {
                useful_exam_types[0], useful_exam_types[1]
            } or set(x["physical_exam_type"]) == {
                useful_exam_types[0], useful_exam_types[2]
            }) and len(x["physical_exam_type"]) == 2)["id"].unique()
        exam_type_filter_df = exam_type_total_df[exam_type_total_df["id"].isin(
            filter_id_list)]

        # Step 2: 通过前后两次对比的数据构造Label
        # 条件一：前后两次体检时员工所在的公司需保持一致
        # 条件二：上岗体检的时间在在岗或离岗体检的时间之前
        logger.info(f"Construct data label")
        mesg_before_df = exam_type_filter_df[
            exam_type_filter_df["physical_exam_type"] == useful_exam_types[0]]
        mesg_after_df = exam_type_filter_df[
            exam_type_filter_df["physical_exam_type"].isin(
                useful_exam_types[1:])]
        mesg_merge_df = pd.merge(left=mesg_before_df,
                                 right=mesg_after_df,
                                 on=["id", "organization_name"],
                                 how="inner",
                                 suffixes=("_before", "_after"))
        mesg_merge_df = mesg_merge_df[
            mesg_merge_df["physical_exam_date_after"] >
            mesg_merge_df["physical_exam_date_before"]]

        # 将诊断的结果打散为键值对，方便进行对比
        mesg_merge_df["physical_exam_conclusion_before_dict"] = mesg_merge_df[
            "physical_exam_conclusion_before"].apply(
                lambda x: x.split(";")).apply(
                    lambda x: seq(x).map(lambda y: y.split("_")).dict())
        mesg_merge_df["physical_exam_conclusion_after_dict"] = mesg_merge_df[
            "physical_exam_conclusion_after"].apply(
                lambda x: x.split(";")).apply(
                    lambda x: seq(x).map(lambda y: y.split("_")).dict())
        # 条件一：后一次检验的结果是前一次子集的，label记为False
        mesg_merge_df["label"] = mesg_merge_df.apply(
            isnot_dict_subset,
            args=("physical_exam_conclusion_before_dict",
                  "physical_exam_conclusion_after_dict"),
            axis=1)
        # 条件二：后一次体检结论中，对各个环境暴露结果均为“目前未见异常”或者“复查”的，label记为False
        row_indexer = mesg_merge_df[
            mesg_merge_df["physical_exam_conclusion_after_dict"].apply(
                lambda x: set([value for value in x.values()]).issubset(
                    {"目前未见异常", "复查"}))].index
        mesg_merge_df.loc[row_indexer, "label"] = False
        # 条件三：后一次体检结论中，与前一次体检结论之间存在差异的结果结论为“目前未见异常”或者“复查”的，label记为False
        row_indexer = mesg_merge_df[mesg_merge_df.apply(
            is_dict_diff_subset,
            args=("physical_exam_conclusion_before_dict",
                  "physical_exam_conclusion_after_dict", {"目前未见异常", "复查"}),
            axis=1)].index
        mesg_merge_df.loc[row_indexer, "label"] = False

        # Step 3: 将打标完毕的数据选取有效字段写入到数据库中
        logger.info("Load labeled data into database")
        valid_cols = [
            col for col in mesg_merge_df.columns if col.endswith("_before")
        ]
        valid_cols += [
            "id", "organization_name", "label", "physical_exam_date_after"
        ]
        labeled_df = mesg_merge_df[valid_cols]
        labeled_df.columns = [
            col[:-7] if col.endswith("_before") else col
            for col in labeled_df.columns
        ]
        labeled_df["duration"] = (
            pd.to_datetime(labeled_df["physical_exam_date_after"]) -
            pd.to_datetime(labeled_df["physical_exam_date"])).dt.days
        labeled_df.drop("physical_exam_date_after", axis=1, inplace=True)

        create_table_from_df(database_path=database_path,
                             dataframe=labeled_df,
                             table_name=labeled_table,
                             primary_keys=["report_card_id"])

    if task == "extract_feature":
        # Step 1: 执行SQL进行所有特征的拼接
        left_table_text = f"SELECT * FROM {labeled_table}"
        right_table_texts = []
        for i in range(len(feature_tables)):
            right_table_texts.append(
                f"\nLEFT JOIN\n(\n\tSELECT * FROM {feature_tables[i]}\n) AS t{i+2}\non t1.report_card_id=t{i+2}.report_card_id"
            )
        total_sql_text = f"""SELECT t1.*,
{",".join([f"t{i}.*" for i in range(2, len(feature_tables)+1)])}
FROM
(
    {left_table_text}
) AS t1{"".join(right_table_texts)}
"""

        logger.info(f"SQL text complete: \n{total_sql_text}")
        logger.info(f"Load data from databsae")
        conn = sl.connect(database_path)
        total_feature_label_df = pd.read_sql_query(sql=total_sql_text,
                                                   con=conn)
        conn.close()

        # Step 2: 去除重复列，去除_unit结尾的列，去除全部为NAN的列
        logger.info(f"Delete columns endwith '_unit'")
        filter_columns = seq(total_feature_label_df.columns).filter(
            lambda x: not x.endswith("_unit"))
        filter_feature_label_df = total_feature_label_df[filter_columns]
        logger.info(f"Delete duplicated columns")
        filter_feature_label_df = filter_feature_label_df.loc[:,
                                                              ~filter_feature_label_df
                                                              .columns.
                                                              duplicated()]

        logger.info(f"Delete NAN(all) columns")
        filter_feature_label_df.replace(to_replace="None",
                                        value=np.nan,
                                        inplace=True)
        filter_feature_label_df.dropna(axis=1, how="all", inplace=True)

        # Step 3: 对剩余的特征进行预览，去除不重要的列
        # 条件一： 非空值不足整体数量的百分之一
        logger.info(f"Delete the columns that count NAN > 99% ")
        for col in filter_feature_label_df.columns:
            logger.info(f"Start to analysis the columns: {col}")
            value_counts = pd.isna(filter_feature_label_df[col]).value_counts()
            if value_counts[False] <= filter_feature_label_df.shape[0] / 100:
                logger.info(
                    f"Drop the columns: {col}, its count NAN = {value_counts[True]}"
                )
                filter_feature_label_df.drop(col, axis=1, inplace=True)

        # Step 4: 将初步遴选完毕数据集的特征与标签写入到数据库中
        # 修改列名中的-, ., ~, (, ), %
        filter_feature_label_df.columns = seq(
            filter_feature_label_df.columns).map(lambda x: re.sub(
                r"-|\.|~|\(|\)", "", x)).map(lambda x: re.sub(r"%", "100", x))
        create_table_from_df(database_path=database_path,
                             dataframe=filter_feature_label_df,
                             table_name=total_feature_label_table,
                             primary_keys=["report_card_id"])

    print(1)
