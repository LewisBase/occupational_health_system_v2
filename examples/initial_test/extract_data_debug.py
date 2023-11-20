# -*- coding: utf-8 -*-
"""
@DATE: 2023-11-17 14:46:26
@Author: Liu Hengjiang
@File: examples\initial_test\extract_data.py
@Software: vscode
@Description:
        进行浙疾控职业健康数据的初步摸查
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


def load_total_data(input_path: str, output_path: str = "./cache", **kwargs):
    nrows = kwargs.get("nrows", None)
    usecols = kwargs.get("usecols", None)
    
    input_path = Path(input_path)
    output_path = Path(output_path)
    if not output_path.exists():
        output_path.mkdir(parents=True)
    df_total = pd.DataFrame()
    if input_path.is_dir():
        for file in input_path.iterdir():
            if file.is_file() and file.suffix == ".xlsx":
                logger.info(f"Load file {file.name}")
                df = pd.read_excel(file, header=0, nrows=nrows, usecols=usecols)
                df_total = pd.concat([df_total, df], axis=0)
    elif input_path.is_file() and input_path.suffix == ".xlsx":
        logger.info(f"Load file {input_path.name}")
        df = pd.read_excel(input_path, header=0, nrows=nrows, usecols=usecols)
        df_total = pd.concat([df_total, df], axis=0)
    else:
        raise ValueError("Invalid Input Path!")

    return df_total


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path",
                        type=str,
                        default="D:\WorkingData\\2023浙江疾控数据\\2022个案卡-20230920\\2022个案卡-1.xlsx")
    args = parser.parse_args()

    logger.info("Input Parameters informations:")
    message = "\n".join([f"{k:<20}: {v}" for k, v in vars(args).items()])
    logger.info(message)

    input_path = args.input_path

    df_total = load_total_data(input_path=input_path, nrows=100, usecols=[0,1,2,3,4,5,6])

    import sqlite3 as sl
    conn = sl.connect("./cache/test.db")
    df_total.to_sql(name="test", con=conn)
    conn.commit()
    conn.close()
    
    

    print(1)
