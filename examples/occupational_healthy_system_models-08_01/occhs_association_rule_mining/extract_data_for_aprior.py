# -*- coding: utf-8 -*-
"""
@DATE: 2024-04-15 10:21:26
@Author: Liu Hengjiang
@File: examples\occhs_association_rule_mining\extract_data_for_aprior.py
@Software: vscode
@Description:
        提取数据进行各种危害暴露及相应各类型职业病的关联关系
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

from utils.database_helper import load_data_from_table

OCCUPATIONAL_DISEASE_TYPE_DICT = {
    "听力": "职业性听力损伤",
    "听阈": "职业性听力损伤",
    "聋": "职业性听力损伤",
    "皮肤": "职业性皮肤病",
    "疹": "职业性皮肤病",
    "血": "职业性心血管系统系统疾病",
    "心脏": "职业性心血管系统系统疾病",
    "心电": "职业性心血管系统系统疾病",
    "呼吸系统": "职业性呼吸系统疾病",
    "肺": "职业性呼吸系统疾病",
    "支气管": "职业性呼吸系统疾病",
    "嗅": "职业性呼吸系统疾病",
    "鼻": "职业性呼吸系统疾病",
    "内分泌": "职业性内分泌系统疾病",
    "甲状腺": "职业性内分泌系统疾病",
    "泌尿": "职业性泌尿生殖系统疾病",
    "糖尿": "职业性泌尿生殖系统疾病",
    "生殖": "职业性泌尿生殖系统疾病",
    "神经系统": "职业性神经系统疾病",
    "周围神经病": "职业性神经系统疾病",
    "视力": "职业性眼病",
    "色": "职业性眼病",
    "盲": "职业性眼病",
    "角膜": "职业性眼病",
    "白内障": "职业性眼病",
    "肝": "职业性中毒性肝病",
    "肾": "职业性中毒性肾病",
    "肿瘤": "职业性肿瘤",
    "放射性": "职业性放射性疾病",
    "骨": "职业性骨关节疾病"
}


def extract_diagnoise_conclusion_split(x):
    if x is not None:
        pair_split = seq(x.split(";")).filter(lambda x: x != "")
        try:
            res = pair_split.map(lambda x: (x.split("_")[0], x.split("_")[1])
                                 if len(x.split("_")) == 2 else None).filter(
                                     lambda x: x is not None).dict()
            return res
        except IndexError:
            logger.error(f"Split error: {x}")
    else:
        return {}


def extract_diagnoise_type_match(row):
    dict1 = row["physical_exam_conclusion_dict"]
    dict2 = row["physical_exam_conclusion_detail_dict"]
    merged_dict = {}
    common_keys = set(dict1.keys()).intersection(dict2.keys())
    for key in common_keys:
        merged_dict.update({key: {dict1[key]: dict2[key]}})
    return merged_dict


def extract_diagnoise_result(x, convert_dict=OCCUPATIONAL_DISEASE_TYPE_DICT):
    res = []
    for key, item in x.items():
        for subkey, subitem in item.items():
            if subkey in ("其他疾病或异常", "疑似职业病", "目前未见异常"):
                pass
            else:
                for dict_key, dict_value in convert_dict.items():
                    if dict_key in subitem:
                        res.append({dict_value: key})
    return seq(seq(res).map(lambda x: tuple(x.items())).flatten().set()).list()



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
        "report_card_id", "physical_exam_conclusion",
        "physical_exam_conclusion_detail",
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
    init_df.dropna(subset=["physical_exam_conclusion"],
                   inplace=True)

    # match and convert the diagnoise conclusion to label
    init_df["physical_exam_conclusion_dict"] = init_df[
        "physical_exam_conclusion"].apply(extract_diagnoise_conclusion_split)
    init_df["physical_exam_conclusion_detail_dict"] = init_df[
        "physical_exam_conclusion_detail"].apply(
            extract_diagnoise_conclusion_split)
    init_df["diagnoise_res"] = init_df.parallel_apply(
        extract_diagnoise_type_match, axis=1)
    init_df["diagnoise_res"] = init_df["diagnoise_res"].apply(extract_diagnoise_result)
    filter_df = init_df[init_df["diagnoise_res"].apply(lambda x: len(x) > 0)][["report_card_id", "diagnoise_res"]].copy()
    filter_df = filter_df.explode("diagnoise_res").reset_index()
    filter_df["disease"] = filter_df["diagnoise_res"].apply(lambda x: x[0])
    filter_df["hazard"] = filter_df["diagnoise_res"].apply(lambda x: x[1])

    # groupby the data
    disease_hazard_groupby_df = pd.DataFrame(filter_df.groupby(["report_card_id", "disease"])["hazard"].apply(lambda x: ",".join(x)))
    disease_hazard_groupby_df.reset_index(inplace=True)
    disease_hazard_groupby_df.to_csv(output_path /
                                "disease_hazard_group_data.csv",
                                header=True,
                                index=False,
                                encoding="utf-8-sig")

    print(1)
