# -*- coding: utf-8 -*-
"""
@DATE: 2024-04-12 11:32:54
@Author: Liu Hengjiang
@File: examples\occhs_multi_label_classification\extract_data_for_mlc.py
@Software: vscode
@Description:
        提取用于构建多标签分类模型的数据
        选取一批职工体检结果样本，根据其体检结论以及体检结论描述对其进行职业病类型的打标
        完成打标后进行多标签分类模型训练
        
        暂不进行特征选择，完成模型构建即可
"""

import re
import pandas as pd
import numpy as np
import sqlite3 as sl
from functional import seq
from loguru import logger
from pandarallel import pandarallel

from utils.database_helper import load_data_from_table, create_table_from_df
from examples.occhs_time_series_predict.extract_data_for_disease_tp import extract_diagnoise_conclusion_split, extract_diagnoise_type_match, OCCUPATIONAL_DISEASE_TYPE_DICT


def extract_diagnoise_result(x, convert_dict=OCCUPATIONAL_DISEASE_TYPE_DICT):
    if x == {}:
        res = ["目前未见异常"]
    else:
        res = []
    for key, item in x.items():
        if key == "其他疾病或异常":
            res.append(key)
        elif key == "疑似职业病":
            res.append(key)
        elif key == "职业禁忌证":
            for dict_key, dict_value in convert_dict.items():
                if dict_key in item:
                    res.append(dict_value)
        else:
            res.append("目前未见异常")
    return ",".join(set(res))


if __name__ == "__main__":
    from datetime import datetime
    logger.add(
        f"./log/extract_data_for_mlc-{datetime.now().strftime('%Y-%m-%d')}.log",
        level="INFO")

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
                        default="BASIC_INFO_AND_MULTILABEL_DEMO")
    parser.add_argument("--feature_tables",
                        type=list,
                        default=[
                            "PHYSI_EXAMI_I", "PHYSI_EXAMI_II",
                            "PHYSI_EXAMI_III", "PHYSI_EXAMI_IV",
                            "PHYSI_EXAMI_V"
                        ])
    parser.add_argument("--total_feature_label_table",
                        type=str,
                        default="TOTAL_FEATURE_AND_MULTILABEL_DEMO")

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

    pandarallel.initialize()

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
        filter_condition = {"is_reexam": "否"}
        limit_size = 0
        logger.info("Load data from database")
        init_df = load_data_from_table(database_path=database_path,
                                       table_name=input_table,
                                       column_names=column_names,
                                       filter_condition=filter_condition,
                                       limit_size=limit_size)

        # 将诊断的结果打散为键值对，方便进行对比
        init_df["physical_exam_conclusion_dict"] = init_df[
            "physical_exam_conclusion"].apply(
                extract_diagnoise_conclusion_split)
        init_df["physical_exam_conclusion_detail_dict"] = init_df[
            "physical_exam_conclusion_detail"].apply(
                extract_diagnoise_conclusion_split)
        init_df["diagnoise_res"] = init_df.parallel_apply(
            extract_diagnoise_type_match, axis=1)
        # Step 1: 进行打标
        init_df["labels"] = init_df["diagnoise_res"].apply(
            extract_diagnoise_result)
        # Step 2: 将打标完毕的数据选取有效字段写入到数据库中
        logger.info("Load labeled data into database")
        valid_cols = [
            col for col in init_df.columns
            if not (col.endswith("_dict") or col.endswith("_res"))
        ]
        labeled_df = pd.DataFrame()  # init_df[valid_cols]
        useful_label = seq(
            init_df["labels"].value_counts().to_dict().items()).filter(
                lambda x: x[1] > 5).map(lambda x: x[0]).list()
        for label in useful_label:
            sub_label_df = init_df[init_df["labels"] ==
                                   label][valid_cols].sample(n=500,
                                                             random_state=42,
                                                             replace=True)
            labeled_df = pd.concat([labeled_df, sub_label_df], axis=0)

        create_table_from_df(database_path=database_path,
                             dataframe=labeled_df,
                             table_name=labeled_table,
                             primary_keys=[])

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
                             primary_keys=[])

    print(1)
