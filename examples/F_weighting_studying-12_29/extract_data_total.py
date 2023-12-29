# -*- coding: utf-8 -*-
"""
@DATE: 2023-12-29 16:40:11
@Author: Liu Hengjiang
@File: examples\F_weighting_studying-12_29\extract_data_total.py
@Software: vscode
@Description:
        提取有效的噪声A计权、C计权数据，探索F计权的构建
"""

import pandas as pd
import pickle
from pathlib import Path
from functional import seq
from joblib import Parallel, delayed
from loguru import logger

from staff_info import StaffInfo


def load_total_data(input_path: Path):
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
                    sub_df_info.loc[:, "duration"] = sub_df_info.loc[:,
                        "duration"].apply(lambda x: 1 if x < 1 else x)

                    # staff_health_info
                    PTA_res_dict = df[[
                        "L-500", "L-1000", "L-2000", "L-3000", "L-4000",
                        "L-6000", "L-8000", "R-500", "R-1000", "R-2000",
                        "R-3000", "R-4000", "R-6000", "R-8000"
                    ]].to_dict(orient="records")
                    PTA_res_dict = seq(PTA_res_dict).map(lambda x: {
                        "PTA": x
                    }).list()
                    sub_df_info.loc[:,"auditory_detection"] = PTA_res_dict

                    # staff_occupational_hazard_info
                    df["recorder_time"] = df["recorder_time"].apply(
                        lambda x: str(x).replace(" ", "").replace(
                            "00:00:00", "").replace(".", "-"))
                    df["parent_path"] = file.parent
                    noise_hazard_dict = df[[
                        "recorder", "recorder_time", "parent_path"
                    ]].to_dict(orient="records")
                    sub_df_info.loc[:,"noise_hazard_info"] = noise_hazard_dict

                    df_total_info = pd.concat([df_total_info, sub_df_info],
                                              axis=0)

    return df_total_info


def _extract_data_for_task(data, task, **additional_set):
    # 加入统一的计算条件设置
    if additional_set is not None:
        data.update(additional_set)
    # 构建对象
    staff_info = StaffInfo(**data)
    return staff_info


def extract_data_for_task(df, task, n_jobs=-1, **additional_set):
    res = Parallel(n_jobs=n_jobs,
                   backend="multiprocessing")(delayed(_extract_data_for_task)(
                       data=data[1].to_dict(), task=task, **additional_set)
                                              for data in df.iterrows())
    return res


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path",
                        type=str,
                        default="D:\WorkingData\工厂噪声数据C计权")
    parser.add_argument("--output_path", type=str, default="./cache")
    parser.add_argument("--additional_set",
                        type=dict,
                        default={
                            "mean_key": [3000, 4000, 6000],
                            "better_ear_strategy": "optimum",
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

    if n_jobs == 1:
        from datetime import datetime
        logger.add(
            f"./log/extract_data_total-{datetime.now().strftime('%Y-%m-%d')}.log",
            level="INFO")

    df_reorganised = load_total_data(input_path=input_path)
    mesg_extract = extract_data_for_task(df=df_reorganised,
                                         n_jobs=n_jobs,
                                         task="standard",
                                         **additional_set)
    pickle.dump(mesg_extract, open(output_path / "extract_data.pkl", "wb"))
    mesg_extract_load = pickle.load(
        open(output_path / "extract_data.pkl", "rb"))
    print(1)
