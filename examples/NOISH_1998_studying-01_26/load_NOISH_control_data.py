# -*- coding: utf-8 -*-
"""
@DATE: 2024-01-26 09:30:21
@Author: Liu Hengjiang
@File: examples\\NOISH_1998_studying-01_26\load_NOISH_control_data.py
@Software: vscode
@Description:
        加载NOISH，1998论文中使用的数据，进行论文内容的复现
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
from joblib import Parallel, delayed

from staff_info import StaffInfo


def work_position_convert(df):
    res = df.plant1.map({
        0: "",
        1: "Steel Fabrication"
    }) + df.plant2.map({
        0: "",
        1: "PAPER BAG MAKING"
    }) + df.plant3.map({
        0: "",
        1: "PRINTING"
    }) + df.plant4.map({
        0: "",
        1: "ALUMINUM FABRIC/PROCESSING"
    }) + df.plant5.map({
        0: "",
        1: "QUARRY"
    }) + df.plant6.map({
        0: "",
        1: "WOODWORKING"
    }) + df.plant7.map({
        0: "",
        1: "TUNNEL PATROL"
    }) + df.plant8.map({
        0: "",
        1: "TRUCKING"
    }) + df.plant9.map({
        0: "",
        1: "HYDROELECTRIC"
    })
    return res


def load_total_data(input_path: Path,
                    sheet_name: str = "NIOSH-data",
                    nrows=2066):
    if input_path.is_file() and input_path.suffix == ".xlsx":
        logger.info(f"Load file {input_path.name}")
        convert_dict = {
            "IScreen": int, "age72": int, "dur72": int,
            "dBA72": int, "Y1234": int, "age1": int,
            "dBA1": int, "plant1": int, "plant2": int,
            "plant3": int, "plant4": int, "plant5": int,
            "plant6": int, "plant7": int, "plant8 ": int,
            "plant9 ": int,
        }
        usecols = seq(range(34)).list() + seq(range(46, 58)).list() + seq(
            range(59, 70)).list()

        col_names = None
        df = pd.read_excel(input_path,
                           sheet_name=sheet_name,
                           header=0,
                           nrows=nrows,
                           usecols=usecols,
                           names=col_names,
                           dtype=convert_dict)
        experiment_df = df.query(
            "IScreen == 1 and Expstat == 'C' and grp == 'CS'")
        experiment_df.rename(columns={
            "ID": "staff_id", "Expyr": "duration", "dBA": "LAeq",
            "L500": "L-500", "L1": "L-1000", "L2": "L-2000",
            "L3": "L-3000", "L4": "L-4000", "L6": "L-6000",
            "R500": "R-500", "R1": "R-1000", "R2": "R-2000",
            "R3": "R-3000", "R4": "R-4000", "R6": "R-6000",
        },inplace=True)

        # staff_basic_info
        experiment_df["work_position"] = work_position_convert(experiment_df)
        sub_df_info = experiment_df[["staff_id", "age", "duration", "work_position"]]

        # staff_health_info
        PTA_res_dict = experiment_df[[
            "L-500", "L-1000", "L-2000", "L-3000", "L-4000", "L-6000",
            "R-500", "R-1000", "R-2000", "R-3000", "R-4000", "R-6000",
        ]].to_dict(orient="records")
        PTA_res_dict = seq(PTA_res_dict).map(lambda x: {"PTA": x}).list()
        sub_df_info["auditory_detection"] = PTA_res_dict

        # staff_occupational_hazard_info
        sub_df_info["noise_hazard_info"] = experiment_df["LAeq"].apply(lambda x: {"LAeq": x})

    return experiment_df, sub_df_info


def _extract_data_for_task(data, **additional_set):
    default_value = {
            "sex": "M",
            "factory_name": "",
            "work_shop": "",
        }
    data.update(default_value)
    # 加入统一的计算条件设置
    if additional_set is not None:
        data.update(additional_set)
    # 构建对象
    staff_info = StaffInfo(**data)
    return staff_info


def extract_data_for_task(df, n_jobs=-1, **additional_set):
    res = Parallel(n_jobs=n_jobs,
                   backend="multiprocessing")(delayed(_extract_data_for_task)(
                       data=data[1].to_dict(), **additional_set)
                                              for data in df.iterrows())
    return res


if __name__ == "__main__":
    from datetime import datetime
    logger.add(
        f"./log/load_NOISH_control_data-{datetime.now().strftime('%Y-%m-%d')}.log",
        level="INFO")

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path",
                        type=str,
                        default="D:\WorkingData\ONHS Data\ONHS_all.xlsx")
    parser.add_argument("--output_path", type=str, default="./cache")
    parser.add_argument("--additional_set",
                        type=dict,
                        default={
                            "mean_key": [1000, 2000, 3000, 4000],
                            "PTA_value_fix": False,
                            "better_ear_strategy": "average_freq",
                            "NIPTS_diagnose_strategy": "better"
                        })
    parser.add_argument("--n_jobs", type=int, default=-1)
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

    df_original, df_extract = load_total_data(input_path=input_path)
    mesg_extract = extract_data_for_task(df=df_extract,
                                         n_jobs=n_jobs,
                                         **additional_set)
    # calculate value test
    HL1234 = seq(mesg_extract).map(lambda x: x.staff_health_info.auditory_detection.get("PTA").mean(mean_key=[1000, 2000, 3000, 4000])).list()
    HL123 = seq(mesg_extract).map(lambda x: x.staff_health_info.auditory_detection.get("PTA").mean(mean_key=[1000, 2000, 3000])).list()
    HL512 = seq(mesg_extract).map(lambda x: x.staff_health_info.auditory_detection.get("PTA").mean(mean_key=[500, 1000, 2000])).list()
    HL346 = seq(mesg_extract).map(lambda x: x.staff_health_info.auditory_detection.get("PTA").mean(mean_key=[3000, 4000, 6000])).list()
    for col, value in zip(("HL1234", "HL123", "HL512", "HL346"), (HL1234, HL123, HL512, HL346)):
        value_array = (df_original[col] - value).unique()
        value_array = seq(value_array).map(lambda x: 0 if x< 1E3 else x).set()
        assert len(value_array) == 1 , f"Calculate results of {col} error!"

    file_suffix = "".join(seq(additional_set["mean_key"]).map(lambda x: str(x)[0]))
    file_suffix = "5" + file_suffix.replace("5", "") if "5" in file_suffix else file_suffix
    pickle.dump(mesg_extract, open(output_path / f"extract_NOISH_control_data-{file_suffix}.pkl",
                                   "wb"))
