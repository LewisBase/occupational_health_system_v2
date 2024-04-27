# -*- coding: utf-8 -*-
"""
@DATE: 2024-04-18 10:33:10
@Author: Liu Hengjiang
@File: examples\CNNMMoE_studying-04_18\extract_Chinese_tensor_data.py
@Software: vscode
@Description:
        以张量形式提取所有中国工人的噪声暴露数据
"""

import re
import pickle
import pandas as pd
import numpy as np
import pickle
from joblib import Parallel, delayed
from pathlib import Path
from functional import seq
from itertools import product
from loguru import logger

import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from sklearn.model_selection import train_test_split

from staff_info import StaffInfo
from diagnose_info.auditory_diagnose import AuditoryDiagnose
from utils.data_helper import root_mean_squared_error


def _extract_data_for_task(data: StaffInfo, **additional_set):  # type: ignore
    mean_key = additional_set.pop("mean_key")
    better_ear_strategy = additional_set.pop("better_ear_strategy")
    NIPTS_diagnose_strategy = additional_set.pop("NIPTS_diagnose_strategy")

    res = {}
    res["staff_id"] = data.staff_id
    # worker information
    # res["sex"] = data.staff_basic_info.sex
    # res["age"] = data.staff_basic_info.age
    # res["duration"] = data.staff_basic_info.duration
    # label information
    res["NIPTS"] = data.staff_health_info.auditory_diagnose.get("NIPTS")
    res["NIPTS_pred_2013"] = data.NIPTS_predict_iso1999_2013(percentrage=50,
                                                             mean_key=mean_key)
    res["NIPTS_pred_2023"] = data.NIPTS_predict_iso1999_2023(percentrage=50, mean_key=mean_key)
    # for freq in [1000, 2000, 3000, 4000, 6000]:
    for freq in [3000, 4000, 6000]:
        res["NIPTS_" + str(freq)] = AuditoryDiagnose.NIPTS(
            detection_result=data.staff_health_info.auditory_detection["PTA"],
            sex=data.staff_basic_info.sex,
            age=data.staff_basic_info.age,
            mean_key=[freq],
            NIPTS_diagnose_strategy=NIPTS_diagnose_strategy)
        res["NIPTS_pred_2013_" + str(freq)] = data.NIPTS_predict_iso1999_2013(percentrage=50,mean_key=[freq])
        res["NIPTS_pred_2023_" + str(freq)] = data.NIPTS_predict_iso1999_2023(percentrage=50,mean_key=[freq])
    # feature information
    ## L
    res["SPL_dB"] = data.staff_occupational_hazard_info.noise_hazard_info.SPL_dB
    res["SPL_dBA"] = data.staff_occupational_hazard_info.noise_hazard_info.SPL_dBA
    # res["SPL_dBC"] = data.staff_occupational_hazard_info.noise_hazard_info.SPL_dBC
    ## kurtosis
    res["kurtosis"] = data.staff_occupational_hazard_info.noise_hazard_info.kurtosis
    # res["A_kurtosis"] = data.staff_occupational_hazard_info.noise_hazard_info.A_kurtosis
    # res["C_kurtosis"] = data.staff_occupational_hazard_info.noise_hazard_info.C_kurtosis
    ## Peak SPL
    res["Peak_SPL_dB"] = data.staff_occupational_hazard_info.noise_hazard_info.Peak_SPL_dB
    ## other features in frequency domain
    for key, value in data.staff_occupational_hazard_info.noise_hazard_info.parameters_from_file.items(
    ):
        if (re.findall(r"\d+",
                       key.split("_")[1])
                if len(key.split("_")) > 1 else False):
            if key.split("_")[0] != "Leq":
                res[key] = value

    return res


def extract_data_for_task(df, n_jobs=-1, **additional_set):
    res = Parallel(n_jobs=n_jobs, backend="multiprocessing")(
        delayed(_extract_data_for_task)(data=data, **additional_set)
        for data in df)
    res_df = pd.DataFrame(res)
    return res_df


class TrainDataSet(Dataset):
    def __init__(self, data, windows_num: int = 400):
        self.feature = []
        for train_features in data[0]:
            tensor_list = []
            for features in train_features:
                tensor_list.append(torch.tensor(features[:windows_num]))
            padded_tensor = pad_sequence([
                torch.cat(
                    (tensor,
                     torch.zeros(windows_num - len(tensor), dtype=torch.long)))
                for tensor in tensor_list
            ],
                                         batch_first=True)
            self.feature.append(padded_tensor)

        self.label = data[1]
        self.sublabel_1 = data[2]
        self.sublabel_2 = data[3]
        self.sublabel_3 = data[4]
        self.iso1999_label = data[5]

    def __getitem__(self, index):
        feature = self.feature[index]
        label = self.label[index]
        sublabel_1 = self.sublabel_1[index]
        sublabel_2 = self.sublabel_2[index]
        sublabel_3 = self.sublabel_3[index]
        iso1999_label = self.iso1999_label[index]
        return feature, label, sublabel_1, sublabel_2, sublabel_3, iso1999_label

    def __len__(self):
        return len(self.feature)


if __name__ == "__main__":
    from datetime import datetime
    logger.add(
        f"./log/extract_Chinese_tensor_data-{datetime.now().strftime('%Y-%m-%d')}.log",
        level="INFO")

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_path",
        type=str,
        default="..//NOISH_1998_studying-01_26//cache//extract_Chinese_data.pkl")
    parser.add_argument("--output_path", type=str, default=".//results")
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

    input_path = Path(args.input_path)
    output_path = Path(args.output_path)
    additional_set = args.additional_set
    n_jobs = args.n_jobs

    # mission start
    ## load tensor data from pkl
    original_data = pickle.load(open(input_path, "rb"))
    original_data = seq(original_data).flatten().list()

    extract_df = extract_data_for_task(df=original_data,
                                       n_jobs=n_jobs,
                                       **additional_set)
    extract_df.index = extract_df["staff_id"]
    extract_df.drop(["staff_id"], axis=1, inplace=True)
    logger.info(f"Size: {extract_df.shape}")
    extract_df.dropna(how="any", axis=0, inplace=True)
    logger.info(f"Size after dropna: {extract_df.shape}")

    train_data, val_data = train_test_split(extract_df,
                                            test_size=0.3,
                                            random_state=42)

    # RMSE test
    ## ISO 1999:2013
    ISO_2013_RMSE = root_mean_squared_error(val_data["NIPTS"], val_data["NIPTS_pred_2013"])
    ISO_2013_3000_RMSE = root_mean_squared_error(val_data["NIPTS_3000"], val_data["NIPTS_pred_2013_3000"])
    ISO_2013_4000_RMSE = root_mean_squared_error(val_data["NIPTS_4000"], val_data["NIPTS_pred_2013_4000"])
    ISO_2013_6000_RMSE = root_mean_squared_error(val_data["NIPTS_6000"], val_data["NIPTS_pred_2013_6000"])
    ## ISO 1999:2023
    ISO_2023_RMSE = root_mean_squared_error(val_data["NIPTS"], val_data["NIPTS_pred_2023"])
    ISO_2023_3000_RMSE = root_mean_squared_error(val_data["NIPTS_3000"], val_data["NIPTS_pred_2023_3000"])
    ISO_2023_4000_RMSE = root_mean_squared_error(val_data["NIPTS_4000"], val_data["NIPTS_pred_2023_4000"])
    ISO_2023_6000_RMSE = root_mean_squared_error(val_data["NIPTS_6000"], val_data["NIPTS_pred_2023_6000"])
                                            
    feature_columns = seq(
        extract_df.columns).filter(lambda x: not x.startswith("NIPTS"))
    train_dataset = TrainDataSet((
        train_data[feature_columns].values,
        train_data["NIPTS"].values,
        train_data["NIPTS_3000"].values,
        train_data["NIPTS_4000"].values,
        train_data["NIPTS_6000"].values,
        train_data["NIPTS_pred_2013"].values,
    ))
    val_dataset = TrainDataSet((
        val_data[feature_columns].values,
        val_data["NIPTS"].values,
        val_data["NIPTS_3000"].values,
        val_data["NIPTS_4000"].values,
        val_data["NIPTS_6000"].values,
        val_data["NIPTS_pred_2013"].values,
    ))
    pickle.dump(train_dataset, open(output_path / "train_dataset.pkl", "wb"))
    pickle.dump(val_dataset, open(output_path / "val_dataset.pkl", "wb"))

    print(1)
