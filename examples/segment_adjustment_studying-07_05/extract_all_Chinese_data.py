# -*- coding: utf-8 -*-
"""
@DATE: 2024-07-05 15:50:22
@Author: Liu Hengjiang
@File: examples\segment_adjustment_studying-07_05\extract_all_Chinese_data.py
@Software: vscode
@Description:
        提取所有中国工人的噪声暴露明细数据
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
from utils.data_helper import root_mean_squared_error, filter_data

general_calculate_func = {
    "arimean": np.mean,
    "median": np.median,
    "geomean": lambda x: 10**(np.mean(np.log10(x))),
}


def _extract_data_for_task(data: StaffInfo, **additional_set):  # type: ignore
    mean_key = additional_set.pop("mean_key")
    better_ear_strategy = additional_set.pop("better_ear_strategy")
    NIPTS_diagnose_strategy = additional_set.pop("NIPTS_diagnose_strategy")
    extrapolation = additional_set.get("extrapolation")
    beta_baseline = additional_set.get("beta_baseline")

    res = {}
    res["staff_id"] = data.staff_id
    # worker information
    # res["name"] = data.staff_basic_info.name
    res["factory_name"] = data.staff_id.split("-")[0]
    res["sex"] = data.staff_basic_info.sex
    res["age"] = data.staff_basic_info.age
    res["duration"] = data.staff_basic_info.duration
    res["work_shop"] = data.staff_basic_info.work_shop
    res["work_position"] = data.staff_basic_info.work_position
    # res["smoking"] = data.staff_basic_info.smoking
    # res["year_of_smoking"] = data.staff_basic_info.year_of_smoking
    PTA_res = data.staff_health_info.auditory_detection.get("PTA")
    res.update(
        seq(PTA_res.data.items()).map(
            lambda x: (x[0], float(x[1]) if re.fullmatch(
                r"-?\d+(\.\d+)?", str(x[1])) else np.nan)).dict())
    # label information
    res["NIPTS"] = data.staff_health_info.auditory_diagnose.get("NIPTS")
    # res["NIPTS_pred_2013"] = data.NIPTS_predict_iso1999_2013(percentrage=50,
    #                                                          mean_key=mean_key)
    # res["NIPTS_pred_2023"] = data.NIPTS_predict_iso1999_2023(percentrage=50,
    #                                                          mean_key=mean_key)
    # feature information
    ## L
    res["LAeq"] = data.staff_occupational_hazard_info.noise_hazard_info.LAeq
    res["SPL_dBA"] = data.staff_occupational_hazard_info.noise_hazard_info.SPL_dBA
    ## kurtosis
    res["kurtosis_arimean"] = data.staff_occupational_hazard_info.noise_hazard_info.kurtosis_arimean
    res["kurtosis_geomean"] = data.staff_occupational_hazard_info.noise_hazard_info.kurtosis_geomean
    res["kurtosis"] = data.staff_occupational_hazard_info.noise_hazard_info.kurtosis
    ## adjust L
    # for method, algorithm_code in product(
    #     ["total_ari", "total_geo", "segment_ari"], ["A+n"]):
    #     res[f"L{algorithm_code[0]}eq_adjust_{method}_{algorithm_code}"] = data.staff_occupational_hazard_info.noise_hazard_info.L_adjust[
    #         method].get(algorithm_code)
    ## NIPTS adjust results
    # res["NIPTS_pred_2013_adjust_ari"] = data.NIPTS_predict_iso1999_2013(
    #     LAeq=res["LAeq_adjust_total_ari_A+n"])
    # res["NIPTS_pred_2023_adjust_ari"] = data.NIPTS_predict_iso1999_2023(
    #     LAeq=res["LAeq_adjust_total_ari_A+n"], extrapolation=extrapolation)
    # res["NIPTS_pred_2013_adjust_geo"] = data.NIPTS_predict_iso1999_2013(
    #     LAeq=res["LAeq_adjust_total_geo_A+n"])
    # res["NIPTS_pred_2023_adjust_geo"] = data.NIPTS_predict_iso1999_2023(
    #     LAeq=res["LAeq_adjust_total_geo_A+n"], extrapolation=extrapolation)
    # res["NIPTS_pred_2013_adjust_segari"] = data.NIPTS_predict_iso1999_2013(
    #     LAeq=res["LAeq_adjust_segment_ari_A+n"])
    # res["NIPTS_pred_2023_adjust_segari"] = data.NIPTS_predict_iso1999_2023(
    #     LAeq=res["LAeq_adjust_segment_ari_A+n"], extrapolation=extrapolation)

    return res


def extract_data_for_task(df, n_jobs=-1, **additional_set):
    res = Parallel(n_jobs=n_jobs, backend="multiprocessing")(
        delayed(_extract_data_for_task)(data=data, **additional_set)
        for data in df)
    res_df = pd.DataFrame(res)
    return res_df


def SPL_pad_func(x: torch.tensor):
    value = 10 * torch.log10(torch.mean(10**(x/10)))
    return value.item()


def kurtosis_pan_func(x: torch.tensor):
    value = 3
    return value

class TrainDataSet(Dataset):

    def __init__(self, data, windows_num: int = 480) -> None:
        self.feature = []
        for train_features in data[0]:
            tensor_list = []
            for features, pad_func in zip(train_features, (SPL_pad_func, kurtosis_pan_func)):
                cut_off_tensor = torch.tensor(features[:windows_num])
                tensor_list.append((cut_off_tensor, pad_func(cut_off_tensor)))
            # !为降低填充值的影响，改为对SPL按有效长度内的等效均值进行填充、kurtosis按高斯噪声峰度值3进行填充
            padded_tensor = pad_sequence([
                torch.cat(
                    (tensor,
                     pad_value * torch.ones(windows_num - len(tensor), dtype=torch.long)))
                for tensor, pad_value in tensor_list
            ],
                                         batch_first=True)
            self.feature.append(padded_tensor)

        self.label = data[1]
        self.other_features = data[2]

    def __getitem__(self, index):
        feature = self.feature[index]
        label = self.label[index]
        other_features = self.other_features[index]
        return feature, label, other_features

    def __len__(self):
        return len(self.feature)


if __name__ == "__main__":
    from datetime import datetime
    logger.add(
        f"./log/extract_all_Chinese_data-{datetime.now().strftime('%Y-%m-%d')}.log",
        level="INFO")

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_path",
        type=str,
        default="../NOISH_1998_studying-01_26/cache/extract_Chinese_data.pkl")
    parser.add_argument("--output_path", type=str, default="./results")
    parser.add_argument("--picture_path", type=str, default="./pictures")
    parser.add_argument("--model_path", type=str, default="./models")
    parser.add_argument("--additional_set",
                        type=dict,
                        default={
                            "mean_key": [3000, 4000, 6000],
                            "better_ear_strategy": "optimum_freq",
                            "NIPTS_diagnose_strategy": "better",
                            "extrapolation": "Linear",
                            "beta_baseline": 3
                        })
    parser.add_argument("--n_jobs", type=int, default=-1)
    parser.add_argument(
        "--annotated_bad_case",
        type=list,
        default=[
            "沃尔夫链条-60",
            "杭州重汽发动机有限公司-10",
            "浙江红旗机械有限公司-6",
            "Wanhao furniture factory-41",
            "Songxia electrical appliance factory-40",
            "Songxia electrical appliance factory-18",
            "Songxia electrical appliance factory-15",
            "Mamibao baby carriage manufactory-77",
            "Liyuan hydroelectric-51",
            "Liyuan hydroelectric-135",
            "Liyuan hydroelectric-112",
            "Liyuan hydroelectric-103",
            "Huahui Machinery-11",
            "Hebang brake pad manufactory-95",
            "Hebang brake pad manufactory-94",
            "Gujia furniture factory-9",
            "Gujia furniture factory-85",
            "Gujia furniture factory-54",
            "Gujia furniture factory-5",
            "Gujia furniture factory-39",
            "Gujia furniture factory-35",
            "Gengde electronic equipment factory-57",
            "Gengde electronic equipment factory-47",
            "Changhua Auto Parts Manufactory-6",
            "Changhua Auto Parts Manufactory-127",
            "Botai furniture manufactory-17",
            "Banglian spandex-123",
            "Changhua Auto Parts Manufactory-40",
            "Banglian spandex-12",
            "Changhua Auto Parts Manufactory-270",
            "Changhua Auto Parts Manufactory-48",
            "Gujia furniture factory-35",
            "Hebang brake pad manufactory-165",
            "Hebang brake pad manufactory-20",
            "Hengfeng paper mill-31",
            "Liyuan hydroelectric-135",
            "Liyuan hydroelectric-30",
            "NSK Precision Machinery Co., Ltd-109",
            "NSK Precision Machinery Co., Ltd-345",
            "Songxia electrical appliance factory-15",
            "Waigaoqiao Shipyard-170",
            "Waigaoqiao Shipyard-94",
            "春风动力-119",
            "浙江红旗机械有限公司-20",
            "浙江红旗机械有限公司-5",
            "Banglian spandex-123",
            "Botai furniture manufactory-66",
            "Changhua Auto Parts Manufactory-120",
            "Changhua Auto Parts Manufactory-141",
            "Changhua Auto Parts Manufactory-355",
            "Changhua Auto Parts Manufactory-40",
            "Gujia furniture factory-39",
            "Gujia furniture factory-5",
            "Gujia furniture factory-85",
            "Hengfeng paper mill-27",
            "Hengjiu Machinery-15",
            "Liyuan hydroelectric-120",
            "Liyuan hydroelectric-14",
            "NSK Precision Machinery Co., Ltd-288",
            "NSK Precision Machinery Co., Ltd-34",
            "Yufeng paper mill-26",
            "春风动力-98",
            "春江-1",
            "东华链条厂-60",
            "东华链条厂-77",
            "东华链条厂-79",
            "双子机械-9",
            "沃尔夫链条-59",
            "中国重汽杭州动力-83",
            "Wanhao furniture factory-24",
            "永创智能-46",
            "Wanhao furniture factory-34",
            "永创智能-45",
            "总装配厂-117",
            "总装配厂-467",
            "东风汽车有限公司商用车车身厂-259",
            "东风汽车紧固件有限公司-405",
            "东风汽车车轮有限公司-16",
            "Huahui Machinery-10",
            "Gujia furniture factory-3",
            # 原来用来修改的一条记录，这里直接去掉
            "东风汽车有限公司商用车车架厂-197",
        ])
    args = parser.parse_args()

    logger.info("Input Parameters informations:")
    message = "\n".join([f"{k:<20}: {v}" for k, v in vars(args).items()])
    logger.info(message)

    input_path = Path(args.input_path)
    output_path = Path(args.output_path)
    picture_path = Path(args.picture_path)
    model_path = Path(args.model_path)
    additional_set = args.additional_set
    n_jobs = args.n_jobs
    annotated_bad_case = args.annotated_bad_case

    for output in (output_path, picture_path, model_path):
        if not output.exists():
            output.mkdir(parents=True)

    original_data = pickle.load(open(input_path, "rb"))
    original_data = seq(original_data).flatten().list()

    extract_df = extract_data_for_task(df=original_data,
                                       n_jobs=n_jobs,
                                       **additional_set)
    filter_df = filter_data(df_total=extract_df,
                            drop_col=None,
                            dropna_set=["NIPTS", "SPL_dBA", "kurtosis"],
                            str_filter_dict={"staff_id": annotated_bad_case},
                            num_filter_dict={
                                "age": {
                                    "up_limit": 60,
                                    "down_limit": 15
                                },
                                "LAeq": {
                                    "up_limit": 100,
                                    "down_limit": 70
                                }
                            },
                            eval_set=None)
    filter_df = filter_df[filter_df["SPL_dBA"].apply(
        lambda x: not any(np.isnan(x)))]
    filter_df = filter_df[filter_df["kurtosis"].apply(lambda x: len(x) != 0)]
    logger.info(f"Data Size after drop []: {filter_df.shape[0]}")
    filter_df.to_csv(output_path / "filter_Chinese_extract_df.csv",
                     header=True,
                     index=False)

    feature_columns = ["SPL_dBA", "kurtosis"]
    other_features = seq(filter_df.columns).filter(
        lambda x: not x in feature_columns + ["NIPTS"])
    train_dataset = TrainDataSet(
        data=(filter_df[feature_columns].values, filter_df["NIPTS"].values,
              filter_df[other_features].to_dict(orient="records")),
        windows_num=480)
    pickle.dump(train_dataset, open(output_path / "train_dataset.pkl", "wb"))
    print(1)
