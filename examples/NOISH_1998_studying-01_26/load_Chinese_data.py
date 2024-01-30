# -*- coding: utf-8 -*-
"""
@DATE: 2024-01-30 16:06:14
@Author: Liu Hengjiang
@File: examples\\NOISH_1998_studying-01_26\load_Chinese_data.py
@Software: vscode
@Description:
        加载所有来自中国工厂的数据，包括东风厂进行匹配后的数据
"""

import pandas as pd
import numpy as np
import pickle
import ast
from pathlib import Path
from functional import seq
from itertools import product
from joblib import Parallel, delayed
from loguru import logger

from staff_info import StaffInfo


def load_total_data_add_0(input_path):
    input_path = Path(input_path)
    df_total_info = pd.DataFrame()
    for subdir in input_path.iterdir():
        if subdir.is_dir():
            for file in subdir.iterdir():
                if file.is_file(
                ) and file.suffix == ".xlsx" and file.name.startswith(
                        subdir.name):
                    logger.info(f"Load file {file.name}")
                    convert_dict = {"recorder": str, "recorder_time": str}
                    usecols = [
                        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 16, 17,
                        19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32
                    ]
                    col_names = [
                        "factory_name", "recorder", "recorder_time", "sex",
                        "age", "duration", "work_shop", "work_position",
                        "work_shedule", "smoking", "year_of_smoking",
                        "cigarette_per_day", "Leq", "LAeq", "kurtosis_arimean",
                        "kurtosis_geomean", "L-500", "L-1000", "L-2000",
                        "L-3000", "L-4000", "L-6000", "L-8000", "R-500",
                        "R-1000", "R-2000", "R-3000", "R-4000", "R-6000",
                        "R-8000"
                    ]
                    df = pd.read_excel(file,
                                       header=0,
                                       usecols=usecols,
                                       names=col_names,
                                       dtype=convert_dict)
                    df["staff_id"] = df[
                        "factory_name"] + "-" + df.index.astype(str)
                    # staff_basic_info
                    sub_df_info = df[[
                        "staff_id", "factory_name", "sex", "age", "duration",
                        "work_shop", "work_position", "work_shedule",
                        "smoking", "year_of_smoking", "cigarette_per_day"
                    ]]
                    sub_df_info.fillna(
                        {
                            "work_shop": "",
                            "work_position": "",
                            "work_shedule": "",
                            "duration": 1,
                            "smoking": 0,
                            "year_of_smoking": 0,
                            "cigarette_per_day": 0
                        },
                        inplace=True)
                    sub_df_info["duration"] = sub_df_info["duration"].apply(
                        lambda x: 1 if x < 1 else x)

                    # staff_health_info
                    PTA_res_dict = df[[
                        "L-500", "L-1000", "L-2000", "L-3000", "L-4000",
                        "L-6000", "L-8000", "R-500", "R-1000", "R-2000",
                        "R-3000", "R-4000", "R-6000", "R-8000"
                    ]].to_dict(orient="records")
                    PTA_res_dict = seq(PTA_res_dict).map(lambda x: {
                        "PTA": x
                    }).list()
                    sub_df_info["auditory_detection"] = PTA_res_dict

                    # staff_occupational_hazard_info
                    df["recorder_time"] = df["recorder_time"].apply(
                        lambda x: str(x).replace(" ", "").replace(
                            "00:00:00", "").replace(".", "-"))
                    df["parent_path"] = file.parent
                    noise_hazard_dict = df[[
                        "recorder", "recorder_time", "parent_path"
                    ]].to_dict(orient="records")
                    sub_df_info["noise_hazard_info"] = noise_hazard_dict

                    df_total_info = pd.concat([df_total_info, sub_df_info],
                                              axis=0)

    return df_total_info


def load_total_data_add_1(input_path):
    file_names = "东风汽车厂匹配数据.xlsx"
    file = Path(input_path) / file_names
    logger.info(f"Load file {file.name}")
    convert_dict = {"recorder": str, "recorder_time": str}
    usecols = [
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 
        12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 
        30, 31, 32, 34
    ]
    col_names = [
        "factory_name",  "sex", "age", "duration", "work_shop", "work_position", "LAeq", "kurtosis_arimean", "kurtosis_geomean", "kurtosis", "SPL_dBA",
        "L-500", "L-1000", "L-2000", "L-3000", "L-4000", "L-6000", "L-8000", "R-500", "R-1000", "R-2000", "R-3000", "R-4000", "R-6000", "R-8000",
        "staff_id", "recorder", "recorder_time", "parent_path"
    ]
    df = pd.read_excel(file,
                       header=0,
                       usecols=usecols,
                       names=col_names,
                       dtype=convert_dict)
    sub_df_info = df[[
        "staff_id", "factory_name", "sex", "age", "duration",
        "work_shop", "work_position", 
    ]]
    sub_df_info.fillna(
        {
            "work_shop": "",
            "work_position": "",
            "duration": 1,
        },
        inplace=True)
    sub_df_info["duration"] = sub_df_info["duration"].apply(
        lambda x: 1 if x < 1 else x)

    PTA_res_dict = df[[
        "L-500", "L-1000", "L-2000", "L-3000", "L-4000",
        "L-6000", "L-8000", "R-500", "R-1000", "R-2000",
        "R-3000", "R-4000", "R-6000", "R-8000"
    ]].to_dict(orient="records")
    PTA_res_dict = seq(PTA_res_dict).map(lambda x: {
        "PTA": x
    }).list()
    sub_df_info["auditory_detection"] = PTA_res_dict

    df["recorder_time"] = df["recorder_time"].apply(
        lambda x: str(x).replace(" ", "").replace(
            "00:00:00", "").replace(".", "-"))
    df["kurtosis"] = df["kurtosis"].apply(lambda x: [np.nan] if x == "[nan]" else ast.literal_eval(x))
    df["SPL_dBA"] = df["SPL_dBA"].apply(lambda x: [np.nan] if x == "[nan]" else ast.literal_eval(x))
    noise_hazard_dict = df[[
        "recorder", "recorder_time", "parent_path",  "kurtosis_arimean",
                "kurtosis_geomean", "LAeq", "kurtosis", "SPL_dBA"
    ]].to_dict(orient="records")
    sub_df_info["noise_hazard_info"] = noise_hazard_dict
    return sub_df_info
    
    
def load_total_data_add_2(input_path):
    file_names = [
        "2011德清噪声项目汇总录入表.xls", "东南网架(2011).xls", "杭钢2011年噪声检测汇总.xls",
        "2012杭钢汇总 - new.xls", "完整杭州个体噪声2010(补).xls", "万达实耐宝(2010).xls",
        "湖州中海石油金州管道有限公司2010年噪声听力数据（全）.xls"
    ]
    input_path = Path(input_path)

    df_total_info = pd.DataFrame()
    for subdir in input_path.iterdir():
        if subdir.is_dir():
            for file in subdir.iterdir():
                if file.is_file() and file.name in file_names:
                    logger.info(f"Load file {file.name}")
                    convert_dict = {"recorder": str, "recorder_time": str}
                    usecols = [
                        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 19, 20, 21, 22,
                        23, 24, 25, 26, 27, 28, 29, 30, 31, 32
                    ]
                    col_names = [
                        "factory_name", "recorder", "recorder_time", "sex",
                        "age", "duration", "work_shop", "work_position",
                        "work_shedule", "smoking", "year_of_smoking",
                        "cigarette_per_day", "L-500", "L-1000", "L-2000",
                        "L-3000", "L-4000", "L-6000", "L-8000", "R-500",
                        "R-1000", "R-2000", "R-3000", "R-4000", "R-6000",
                        "R-8000"
                    ]
                    df = pd.read_excel(file,
                                       header=0,
                                       usecols=usecols,
                                       names=col_names,
                                       dtype=convert_dict)
                    df["staff_id"] = df[
                        "factory_name"] + "-" + df.index.astype(str)
                    sub_df_info = df[[
                        "staff_id", "factory_name", "sex", "age", "duration",
                        "work_shop", "work_position", "work_shedule",
                        "smoking", "year_of_smoking", "cigarette_per_day"
                    ]]
                    sub_df_info.fillna(
                        {
                            "work_shop": "",
                            "work_position": "",
                            "work_shedule": "",
                            "duration": 1,
                            "smoking": 0,
                            "year_of_smoking": 0,
                            "cigarette_per_day": 0
                        },
                        inplace=True)
                    sub_df_info["duration"] = sub_df_info["duration"].apply(
                        lambda x: 1 if x < 1 else x)
                    PTA_res_dict = df[[
                        "L-500", "L-1000", "L-2000", "L-3000", "L-4000",
                        "L-6000", "L-8000", "R-500", "R-1000", "R-2000",
                        "R-3000", "R-4000", "R-6000", "R-8000"
                    ]].to_dict(orient="records")
                    PTA_res_dict = seq(PTA_res_dict).map(lambda x: {
                        "PTA": x
                    }).list()
                    sub_df_info["auditory_detection"] = PTA_res_dict

                    df["recorder_time"] = df["recorder_time"].apply(
                        lambda x: str(x).replace(" ", "").replace(
                            "00:00:00", "").replace(".", "-"))
                    df["parent_path"] = file.parent
                    noise_hazard_dict = df[[
                        "recorder", "recorder_time", "parent_path"
                    ]].to_dict(orient="records")
                    # sub_df_info["parent_path"] = file.parent
                    sub_df_info["noise_hazard_info"] = noise_hazard_dict

                    df_total_info = pd.concat([df_total_info, sub_df_info],
                                              axis=0)
    return df_total_info


def load_total_data_add_3(input_path):
    file_names = [
        "春风动力(use this one）.xls", "东华链条厂（use this one）.xlsx",
        "力达纺织（use this one）.xlsx", "双子机械（use this one）.xlsx",
        "天地数码（use this one）.xlsx", "万通智能（use this one）.xlsx",
        "沃尔夫链条厂(use s one).xlsx", "浙江春江轻纺数据大全（use this one）.xlsx"
    ]
    input_path = Path(input_path)

    df_total_info = pd.DataFrame()
    for file in input_path.iterdir():
        if file.is_file() and file.name in file_names:
            logger.info(f"Load file {file.name}")
            convert_dict = {
                "recorder": str,
                "recorder_time": str,
                "work_shop": str
            }
            usecols = [
                2, 3, 4, 5, 6, 7, 8, 9, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21,
                22, 23, 24, 25, 26, 27, 28
            ]
            col_names = [
                "sex", "age", "duration", "factory_name", "work_shop",
                "work_position", "recorder_time", "recorder", "LAeq",
                "kurtosis_arimean", "kurtosis_geomean", "L-500", "L-1000",
                "L-2000", "L-3000", "L-4000", "L-6000", "L-8000", "R-500",
                "R-1000", "R-2000", "R-3000", "R-4000", "R-6000", "R-8000"
            ]
            df = pd.read_excel(file,
                               header=0,
                               usecols=usecols,
                               names=col_names,
                               dtype=convert_dict)
            df["staff_id"] = df["factory_name"] + "-" + df.index.astype(str)
            df[["recorder",
                "recorder_time"]] = df[["recorder",
                                        "recorder_time"]].fillna("")
            sub_df_info = df[[
                "staff_id", "factory_name", "sex", "age", "duration",
                "work_shop", "work_position"
            ]]
            sub_df_info[["work_shop", "work_position"
                         ]] = sub_df_info[["work_shop",
                                           "work_position"]].fillna("")
            sub_df_info["duration"] = sub_df_info["duration"].fillna(1)
            sub_df_info["duration"] = sub_df_info["duration"].apply(
                lambda x: 1 if x < 1 else x)

            PTA_res_dict = df[[
                "L-500", "L-1000", "L-2000", "L-3000", "L-4000", "L-6000",
                "L-8000", "R-500", "R-1000", "R-2000", "R-3000", "R-4000",
                "R-6000", "R-8000"
            ]].to_dict(orient="records")
            PTA_res_dict = seq(PTA_res_dict).map(lambda x: {"PTA": x}).list()
            sub_df_info["auditory_detection"] = PTA_res_dict

            noise_hazard_dict = df[[
                "recorder", "recorder_time", "kurtosis_arimean",
                "kurtosis_geomean", "LAeq"
            ]].to_dict(orient="records")
            sub_df_info["noise_hazard_info"] = noise_hazard_dict

            df_total_info = pd.concat([df_total_info, sub_df_info], axis=0)
    return df_total_info


def load_total_data_add_4(input_path):
    file_names = [
        "萧山丝化印染(use this one）.xls", "永创智能 11-30 (use this one).xls",
        "中国重汽（Use this one).xls"
    ]
    input_path = Path(input_path)

    df_total_info = pd.DataFrame()
    for subdir in input_path.iterdir():
        if subdir.is_dir():
            for file in subdir.iterdir():
                if file.is_file() and file.name in file_names:
                    logger.info(f"Load file {file.name}")
                    convert_dict = {
                        "recorder": str,
                        "recorder_time": str,
                        "work_shop": str
                    }
                    usecols = [
                        2, 3, 4, 5, 6, 7, 8, 9, 11, 13, 14, 15, 16, 17, 18, 19,
                        20, 21, 22, 23, 24, 25, 26, 27, 28
                    ]
                    col_names = [
                        "sex", "age", "duration", "factory_name", "work_shop",
                        "work_position", "recorder_time", "recorder", "LAeq",
                        "kurtosis_arimean", "kurtosis_geomean", "L-500",
                        "L-1000", "L-2000", "L-3000", "L-4000", "L-6000",
                        "L-8000", "R-500", "R-1000", "R-2000", "R-3000",
                        "R-4000", "R-6000", "R-8000"
                    ]
                    df = pd.read_excel(file,
                                       header=0,
                                       usecols=usecols,
                                       names=col_names,
                                       dtype=convert_dict)
                    df["staff_id"] = df[
                        "factory_name"] + "-" + df.index.astype(str)
                    df[["recorder",
                        "recorder_time"]] = df[["recorder",
                                                "recorder_time"]].fillna("")
                    sub_df_info = df[[
                        "staff_id", "factory_name", "sex", "age", "duration",
                        "work_shop", "work_position"
                    ]]
                    sub_df_info[["work_shop", "work_position"]] = sub_df_info[[
                        "work_shop", "work_position"
                    ]].fillna("")
                    sub_df_info["duration"] = sub_df_info["duration"].fillna(1)
                    sub_df_info["duration"] = sub_df_info["duration"].apply(
                        lambda x: 1 if x < 1 else x)

                    PTA_res_dict = df[[
                        "L-500", "L-1000", "L-2000", "L-3000", "L-4000",
                        "L-6000", "L-8000", "R-500", "R-1000", "R-2000",
                        "R-3000", "R-4000", "R-6000", "R-8000"
                    ]].to_dict(orient="records")
                    PTA_res_dict = seq(PTA_res_dict).map(lambda x: {
                        "PTA": x
                    }).list()
                    sub_df_info["auditory_detection"] = PTA_res_dict

                    df["recorder_time"] = df["recorder_time"].apply(
                        lambda x: str(x).replace(" ", "").replace(
                            "00:00:00", "").replace(".", "-"))
                    df["parent_path"] = file.parent
                    noise_hazard_dict = df[[
                        "recorder", "recorder_time", "parent_path", "LAeq",
                        "kurtosis_arimean", "kurtosis_geomean"
                    ]].to_dict(orient="records")
                    sub_df_info["noise_hazard_info"] = noise_hazard_dict

                    df_total_info = pd.concat([df_total_info, sub_df_info],
                                              axis=0)
    return df_total_info


def _extract_data_for_task(data, task, **additional_set):
    # 加入统一的计算条件设置
    if additional_set is not None:
        data.update(additional_set)
    
    if task == "standard":
        # 构建对象
        staff_info = StaffInfo(**data)
        try:
            staff_info.staff_occupational_hazard_info.noise_hazard_info = \
                staff_info.staff_occupational_hazard_info.noise_hazard_info. \
                    load_from_preprocessed_file(
                parent_path=data["noise_hazard_info"]["parent_path"],
                recorder=staff_info.staff_occupational_hazard_info.
                noise_hazard_info.recorder,
                recorder_time=staff_info.staff_occupational_hazard_info.
                noise_hazard_info.recorder_time,
                file_name="Kurtosis_Leq.xls",
                sheet_name_prefix="Win_width=40",
                usecols=None,
                col_names=[
                    "FK_63", "FK_125", "FK_250", "FK_500", "FK_1000", "FK_2000",
                    "FK_4000", "FK_8000", "FK_16000", "kurtosis", "SPL_63",
                    "SPL_125", "SPL_250", "SPL_500", "SPL_1000", "SPL_2000",
                    "SPL_4000", "SPL_8000", "SPL_16000", "SPL_dB", "SPL_dBA",
                    "Peak_SPL_dB", "Leq", "LAeq", "kurtosis_median",
                    "kurtosis_arimean", "kurtosis_geomean", "Max_Peak_SPL_dB",
                    "Leq_63", "Leq_125", "Leq_250", "Leq_500", "Leq_1000",
                    "Leq_2000", "Leq_4000", "Leq_8000", "Leq_16000"
                ])
        except FileNotFoundError:
            logger.warning(
                f"Default filename for {staff_info.staff_id} cannot be found, filename Kurtosis_Leq-old.xls used!!!"
            )
            staff_info.staff_occupational_hazard_info.noise_hazard_info = \
                staff_info.staff_occupational_hazard_info.noise_hazard_info.\
                    load_from_preprocessed_file(
                parent_path=data["noise_hazard_info"]["parent_path"],
                recorder=staff_info.staff_occupational_hazard_info.
                noise_hazard_info.recorder,
                recorder_time=staff_info.staff_occupational_hazard_info.
                noise_hazard_info.recorder_time,
                file_name="Kurtosis_Leq-old.xls",
                sheet_name_prefix="Win_width=40",
                usecols=None,
                col_names=[
                    "FK_63", "FK_125", "FK_250", "FK_500", "FK_1000", "FK_2000",
                    "FK_4000", "FK_8000", "FK_16000", "kurtosis", "SPL_63",
                    "SPL_125", "SPL_250", "SPL_500", "SPL_1000", "SPL_2000",
                    "SPL_4000", "SPL_8000", "SPL_16000", "SPL_dB", "SPL_dBA",
                    "Peak_SPL_dB", "Leq", "LAeq", "kurtosis_median",
                    "kurtosis_arimean", "kurtosis_geomean", "Max_Peak_SPL_dB",
                    "Leq_63", "Leq_125", "Leq_250", "Leq_500", "Leq_1000",
                    "Leq_2000", "Leq_4000", "Leq_8000", "Leq_16000"
                ])
    
    if task == "additional":
        noise_file_path = Path(data["noise_hazard_info"]["parent_path"])
        prefix = data["noise_hazard_info"]["recorder_time"] + \
            "-" + data["noise_hazard_info"]["recorder"] + "-"
        # kurtosis文件并非标准格式！
        for file in noise_file_path.iterdir():
            if file.name.startswith(prefix):
                xls = pd.ExcelFile(file)
                valid_sheet_names = seq(xls.sheet_names).filter(
                    lambda x: x.startswith("Win_width=40")).list()
                if len(valid_sheet_names) > 1:
                    raise ValueError("Too many valid sheet in File!")
                if len(valid_sheet_names) == 0:
                    raise ValueError("No valid sheet in File")
                sheet_name = valid_sheet_names[0]
                logger.info(f"Load Noise File {file.name}-{sheet_name}")
                noise_df = pd.read_excel(file, sheet_name=sheet_name, header=0)
                data["noise_hazard_info"]["kurtosis"] = noise_df["Kurtosis"].values.tolist()
                data["noise_hazard_info"]["SPL_dBA"] = noise_df["SPL_dBA"].values.tolist()
                data["noise_hazard_info"]["LAeq"] = noise_df["LAeq"].unique().tolist()[0]
                break
        staff_info = StaffInfo(**data)
        
    if task == "preprocessed":
        staff_info = StaffInfo(**data)
    
    for method, algorithm_code in product(
        ["total_ari", "total_geo", "segment_ari"], ["A+n"]):
        staff_info.staff_occupational_hazard_info.noise_hazard_info.cal_adjust_L(
            method=method, algorithm_code=algorithm_code)
    return staff_info


def extract_data_for_task(df, task, n_jobs=-1, **additional_set):
    res = Parallel(n_jobs=n_jobs,
                   backend="multiprocessing")(delayed(_extract_data_for_task)(
                       data=data[1].to_dict(), task=task, **additional_set)
                                              for data in df.iterrows())
    return res


if __name__ == "__main__":
    from datetime import datetime
    logger.add(f"./log/load_Chinese_data-{datetime.now().strftime('%Y-%m-%d')}.log",level="INFO")
    
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_paths",
                        type=str,
                        default=[
                            "F:\\Mine\\之江实验室\\PostDoc\\2_课题\\WorkingData\\工厂噪声数据",
                            "F:\\Mine\\之江实验室\\PostDoc\\2_课题\WorkingData\\东风汽车制造厂数据",
                            "F:\\Mine\\之江实验室\\PostDoc\\2_课题\WorkingData\\工厂噪声数据-Additional",
                            "F:\\Mine\\之江实验室\\PostDoc\\2_课题\WorkingData\\2021噪声数据-耳蜗突触病",
                            "F:\\Mine\\之江实验室\\PostDoc\\2_课题\WorkingData\\2021噪声数据-耳蜗突触病\含噪声数据"
                        ])
    # parser.add_argument("--input_paths",
    #                     type=str,
    #                     default=[
    #                         # "D:\WorkingData\工厂噪声数据",
    #                         "D:\WorkingData\东风汽车制造厂数据",
    #                         "D:\WorkingData\工厂噪声数据-Additional",
    #                         "D:\WorkingData\\2021噪声数据-耳蜗突触病",
    #                         "D:\WorkingData\\2021噪声数据-耳蜗突触病\含噪声数据"
    #                     ])
    parser.add_argument("--output_path", type=str, default="./cache")
    parser.add_argument("--additional_set",
                        type=dict,
                        default={
                            "mean_key": [3000, 4000, 6000],
                            "better_ear_strategy": "optimum_freq",
                            "NIPTS_diagnose_strategy": "better"
                        })
    parser.add_argument("--n_jobs", type=int, default=-1)
    args = parser.parse_args()

    logger.info("Input Parameters informations:")
    message = "\n".join([f"{k:<20}: {v}" for k, v in vars(args).items()])
    logger.info(message)

    input_paths = args.input_paths
    output_path = Path(args.output_path)
    additional_set = args.additional_set
    n_jobs = args.n_jobs
    if not output_path.exists():
        output_path.mkdir(parents=True)

    total_mesg_extract = []
    load_funcs = (
                  load_total_data_add_0,
                  load_total_data_add_1,
                  load_total_data_add_2,
                  load_total_data_add_3,
                  load_total_data_add_4
                  )
    task_types = (
        "standard",
        "preprocessed",
        "additional",
        "preprocessed",
        "standard"
        )
    
    for load_func, input_path, task in zip(load_funcs, input_paths, task_types):
        df_reorganised = load_func(input_path=input_path)
        mesg_extract = extract_data_for_task(df=df_reorganised,
                                             n_jobs=n_jobs,
                                             task=task,
                                             **additional_set)
        total_mesg_extract.append(mesg_extract)
    pickle.dump(total_mesg_extract, open(output_path / "extract_Chinese_data.pkl",
                                         "wb"))
    mesg_extract_load = pickle.load(
        open(output_path / "extract_Chinese_data.pkl", "rb"))
    mesg_extract_load = seq(mesg_extract_load).flatten().list()
    print(1)
