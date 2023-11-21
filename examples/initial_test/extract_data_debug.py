# -*- coding: utf-8 -*-
"""
@DATE: 2023-11-17 14:46:26
@Author: Liu Hengjiang
@File: examples\initial_test\extract_data.py
@Software: vscode
@Description:
        进行浙疾控职业健康数据的初步摸查
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sqlite3 as sl
from pathlib import Path
from functional import seq
from loguru import logger
from joblib import Parallel, delayed

from staff_info import StaffInfo
from constants.global_constants import PhysicalExamItems


def add_staff_file_to_db(file, output_path, **kwargs):
    nrows = kwargs.get("nrows", None)
    usecols = kwargs.get("usecols", None)
    

    logger.info(f"Load file {file.name}")
    df = pd.read_excel(file, header=0, nrows=nrows, usecols=usecols)
    df.columns = seq(df.columns).map(
        lambda x: PhysicalExamItems.COLUMNS_NAME_DICT.get(x, x)).list()
    for key, value in PhysicalExamItems.COLUMNS_TYPE_DICT.items():
        if key == "BASIC_INFO":
            sub_columns = seq(df.columns).filter(lambda x: x in value).list()
            sub_columns += ["id"]
        else:
            sub_columns = seq(
                df.columns).filter(lambda x: x.split("_")[0] in value).list()
            sub_columns += ["id", "report_card_id", "physical_exam_id"]
        sub_df = df[sub_columns]
        try:
            logger.info(f"Load completed, start to write {file.name} to sqlite")
            con = sl.connect(output_path / "occupational_health_2023.db")
            sub_df.to_sql(name=key, con=con, if_exists="append", index=False)
            con.commit()
            con.close()
        except Exception as e:
            logger.error(f"Error: {e}")


def load_staff_data_to_db(input_path: str,
                          output_path: str = "./cache",
                          n_jobs=1,
                          **kwargs):
    input_path = Path(input_path)
    output_path = Path(output_path)
    if not output_path.exists():
        output_path.mkdir(parents=True)
    if input_path.is_dir():
        input_files = [file for file in input_path.iterdir() if file.is_file() and file.suffix == ".xlsx"]
        Parallel(n_jobs=n_jobs, backend="multiprocessing")(
            delayed(add_staff_file_to_db)(file=file, output_path=output_path, **kwargs) for file in input_files)
    elif input_path.is_file() and input_path.suffix == ".xlsx":
        add_staff_file_to_db(file=input_path, output_path=output_path, **kwargs)
    else:
        raise ValueError("Invalid Input Path!")



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path",
                        type=str,
                        default="D:\WorkingData\\2023浙江疾控数据\parallel_test")
                        # default="D:\WorkingData\\2023浙江疾控数据\\2022个案卡-20230920")
    parser.add_argument("--output_path",
                        type=str,
                        default="D:\WorkingData\\2023浙江疾控数据\DataBase")
    parser.add_argument("--n_jobs",
                        type=int,
                        default=5)
    args = parser.parse_args()

    logger.info("Input Parameters informations:")
    message = "\n".join([f"{k:<20}: {v}" for k, v in vars(args).items()])
    logger.info(message)

    input_path = args.input_path
    output_path = args.output_path
    n_jobs = args.n_jobs

    df_total = load_staff_data_to_db(input_path=input_path,
                                     output_path=output_path,
                                     n_jobs=n_jobs)  #, nrows=100)
    print(1)
