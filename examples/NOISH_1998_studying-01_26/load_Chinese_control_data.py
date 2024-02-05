# -*- coding: utf-8 -*-
"""
@DATE: 2024-02-02 16:03:46
@Author: Liu Hengjiang
@File: examples\\NOISH_1998_studying-01_26\load_Chinese_control_data.py
@Software: vscode
@Description:
        加载收集自中国的对照组数据
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


def load_total_control_data(input_path):
    file = Path(input_path)
    if file.is_file() and file.suffix == ".xlsx":
        logger.info(f"Load file {file.name}")
        col_names = [
            "staff_id", "factory_name", "sex", "age", "duration",
            "L-500", "L-1000", "L-2000","L-3000", "L-4000", "L-6000", "L-8000",
            "R-500", "R-1000", "R-2000", "R-3000", "R-4000", "R-6000", "R-8000"
        ]
        df = pd.read_excel(file,
                           header=0,
                           names=col_names,
                           sheet_name="Control")
        # staff_basic_info
        sub_df_info = df[[
            "staff_id", "factory_name", "sex", "age", "duration",
        ]]
        sub_df_info[["staff_id", "factory_name"]] = sub_df_info[["staff_id", "factory_name"]].astype(str)
        sub_df_info.fillna({"duration": 1}, inplace=True)
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

    return sub_df_info


def _extract_data_for_task(data, task, **additional_set):
    # 加入统一的计算条件设置
    if additional_set is not None:
        data.update(additional_set)
    data.update({"work_shop": "", "work_position":""})
    if task == "control":
        staff_info = StaffInfo(**data)
    
    return staff_info


def extract_data_for_task(df, task, n_jobs=-1, **additional_set):
    res = Parallel(n_jobs=n_jobs,
                   backend="multiprocessing")(delayed(_extract_data_for_task)(
                       data=data[1].to_dict(), task=task, **additional_set)
                                              for data in df.iterrows())
    return res


if __name__ == "__main__":
    from datetime import datetime
    logger.add(f"./log/load_Chinese_control_data-{datetime.now().strftime('%Y-%m-%d')}.log",level="INFO")
    
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path",
                        type=str,
                        default=
                            "D:\WorkingData\Chinese Control Data\Control data.xlsx",
                        )
    parser.add_argument("--output_path", type=str, default="./cache")
    parser.add_argument("--additional_set",
                        type=dict,
                        default={
                            "mean_key": [3000, 4000, 6000],
                            "better_ear_strategy": "optimum_freq",
                            "NIPTS_diagnose_strategy": "better"
                        })
    parser.add_argument("--n_jobs", type=int, default=1)
    args = parser.parse_args()

    logger.info("Input Parameters informations:")
    message = "\n".join([f"{k:<20}: {v}" for k, v in vars(args).items()])
    logger.info(message)

    input_path = Path(args.input_path)
    output_path = Path(args.output_path)
    additional_set = args.additional_set
    n_jobs = args.n_jobs
    if not output_path.exists():
        output_path.mkdir(parents=True)

    
    df_reorganised = load_total_control_data(input_path=input_path)
    mesg_extract = extract_data_for_task(df=df_reorganised,
                                         n_jobs=n_jobs,
                                         task="control",
                                         **additional_set)
    pickle.dump(mesg_extract, open(output_path / "extract_Chinese_control_data.pkl",
                                         "wb"))
    mesg_extract_load = pickle.load(
        open(output_path / "extract_Chinese_control_data.pkl", "rb"))
    print(1)
