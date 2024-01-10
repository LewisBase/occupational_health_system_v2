# -*- coding: utf-8 -*-
"""
@DATE: 2024-01-10 15:30:19
@Author: Liu Hengjiang
@File: examples\MMoE_studying-01_09\MMoE_regression.py
@Software: vscode
@Description:
        使用MMoE模型进行NIPTS回归尝试
"""

import re
import pickle
import pandas as pd
import numpy as np
import statsmodels.api as sm
import pickle
from joblib import Parallel, delayed
from pathlib import Path
from functional import seq
from itertools import product
from loguru import logger
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

from staff_info import StaffInfo
from diagnose_info.auditory_diagnose import AuditoryDiagnose
from model.multi_task.mmoe import MMoE
from utils.data_helper import root_mean_squared_error
from utils.plot_helper import plot_corr_hotmap, plot_feature_importance

from matplotlib.font_manager import FontProperties
from matplotlib import rcParams

config = {
    "font.family": "serif",
    "font.size": 12,
    "mathtext.fontset": "stix",  # matplotlib渲染数学字体时使用的字体，和Times New Roman差别不大
    "font.serif": ["STZhongsong"],  # 华文中宋
    "axes.unicode_minus": False  # 处理负号，即-号
}
rcParams.update(config)


class TrainDataSet(Dataset):
    def __init__(self, data):
        self.feature = data[0]
        self.label1 = data[1]
        self.label2 = data[2]
        self.label3 = data[3]

    def __getitem__(self, index):
        feature = self.feature[index]
        label1 = self.label1[index]
        label2 = self.label2[index]
        label3 = self.label3[index]
        return feature, label1, label2, label3

    def __len__(self):
        return len(self.feature)


general_calculate_func = {
    "arimean": np.mean,
    "median": np.median,
    "geomean": lambda x: 10**(np.mean(np.log10(x))),
}


def _extract_data_for_task(data: StaffInfo, **additional_set):
    better_ear_strategy = additional_set.pop("better_ear_strategy")
    NIPTS_diagnose_strategy = additional_set.pop("NIPTS_diagnose_strategy")

    res = {}
    res["staff_id"] = data.staff_id
    # label information
    res["NIPTS"] = data.staff_health_info.auditory_diagnose.get("NIPTS")
    for freq in [3000, 4000, 6000]:
        res["NIPTS_" + str(freq)] = AuditoryDiagnose.NIPTS(
            detection_result=data.staff_health_info.auditory_detection["PTA"],
            sex=data.staff_basic_info.sex,
            age=data.staff_basic_info.age,
            mean_key=[freq],
            NIPTS_diagnose_strategy=NIPTS_diagnose_strategy)
    # feature information
    ## user features
    res["age"] = data.staff_basic_info.age
    res["sex"] = data.staff_basic_info.sex
    res["duration"] = data.staff_basic_info.duration
    res["work_position"] = data.staff_basic_info.work_position
    ## L
    res["Leq"] = data.staff_occupational_hazard_info.noise_hazard_info.Leq
    res["LAeq"] = data.staff_occupational_hazard_info.noise_hazard_info.LAeq
    ## adjust L
    for method, algorithm_code in product(
        ["total_ari", "total_geo", "segment_ari"], ["A+n"]):
        res[f"L{algorithm_code[0]}eq_adjust_{method}_{algorithm_code}"] = data.staff_occupational_hazard_info.noise_hazard_info.L_adjust[
            method].get(algorithm_code)
    ## kurtosis
    res["kurtosis_geomean"] = data.staff_occupational_hazard_info.noise_hazard_info.kurtosis_geomean
    ## Peak SPL
    res["max_Peak_SPL_dB"] = data.staff_occupational_hazard_info.noise_hazard_info.Max_Peak_SPL_dB
    ## other features in frequency domain
    for key, value in data.staff_occupational_hazard_info.noise_hazard_info.parameters_from_file.items(
    ):
        if (re.findall(r"\d+",
                       key.split("_")[1])
                if len(key.split("_")) > 1 else False):
            if key.split("_")[0] != "Leq":
                for func_name, func in general_calculate_func.items():
                    res[key + "_" + func_name] = func(value)
            else:
                res[key] = value

    return res


def extract_data_for_task(df, n_jobs=-1, **additional_set):
    res = Parallel(n_jobs=n_jobs, backend="multiprocessing")(
        delayed(_extract_data_for_task)(data=data, **additional_set)
        for data in df)
    res_df = pd.DataFrame(res)
    return res_df


def train_model(model, train_loader, val_loader, epoch, loss_function,
                optimizer, path, early_stop):
    """
    pytorch model train function
    :param model: pytorch model
    :param train_loader: dataloader, train data loader
    :param val_loader: dataloader, val data loader
    :param epoch: int, number of iters
    :param loss_function: loss function of train model
    :param optimizer: pytorch optimizer
    :param path: save path
    :param early_stop: int, early stop number
    :return: None
    """
    # use GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # 多少步内验证集的loss没有变小就提前停止
    patience, eval_loss = 0, 0

    # train
    for i in range(epoch):
        y_train_NIPTS_3000_true = []
        y_train_NIPTS_3000_predict = []
        y_train_NIPTS_4000_true = []
        y_train_NIPTS_4000_predict = []
        y_train_NIPTS_6000_true = []
        y_train_NIPTS_6000_predict = []
        total_loss, count = 0, 0
        for idx, (x, y1, y2, y3) in tqdm(enumerate(train_loader),
                                         total=len(train_loader)):
            x, y1, y2, y3 = x.to(device), y1.to(device), y2.to(device), y3.to(
                device)
            predict = model(x)
            y_train_NIPTS_3000_true += list(y1.squeeze().cpu().numpy())
            y_train_NIPTS_4000_true += list(y2.squeeze().cpu().numpy())
            y_train_NIPTS_6000_true += list(y3.squeeze().cpu().numpy())
            y_train_NIPTS_3000_predict += list(
                predict[0].squeeze().cpu().detach().numpy())
            y_train_NIPTS_4000_predict += list(
                predict[1].squeeze().cpu().detach().numpy())
            y_train_NIPTS_6000_predict += list(
                predict[2].squeeze().cpu().detach().numpy())
            loss_1 = loss_function(predict[0], y1.unsqueeze(1).float())
            loss_2 = loss_function(predict[1], y2.unsqueeze(1).float())
            loss_3 = loss_function(predict[2], y3.unsqueeze(1).float())
            loss = loss_1 + loss_2 + loss_3
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += float(loss)
            count += 1
        torch.save(model, path.format(i + 1))
        NIPTS_3000_RMSE = root_mean_squared_error(y_train_NIPTS_3000_true,
                                                  y_train_NIPTS_3000_predict)
        NIPTS_4000_RMSE = root_mean_squared_error(y_train_NIPTS_4000_true,
                                                  y_train_NIPTS_4000_predict)
        NIPTS_6000_RMSE = root_mean_squared_error(y_train_NIPTS_6000_true,
                                                  y_train_NIPTS_6000_predict)
        logger.info(f"Epoch {i} train loss is {round(total_loss/count, 2)}")
        logger.info(
            f"          NIPTS_3000's RMSE is {round(NIPTS_3000_RMSE, 2)}")
        logger.info(
            f"          NIPTS_4000's RMSE is {round(NIPTS_4000_RMSE, 2)}")
        logger.info(
            f"          NIPTS_6000's RMSE is {round(NIPTS_6000_RMSE, 2)}")

        # 验证
        total_eval_loss = 0
        model.eval()
        count_eval = 0
        y_val_NIPTS_3000_true = []
        y_val_NIPTS_4000_true = []
        y_val_NIPTS_6000_true = []
        y_val_NIPTS_3000_predict = []
        y_val_NIPTS_4000_predict = []
        y_val_NIPTS_6000_predict = []
        for idx, (x, y1, y2, y3) in tqdm(enumerate(val_loader),
                                         total=len(val_loader)):
            x, y1, y2, y3 = x.to(device), y1.to(device), y2.to(device), y3.to(
                device)
            predict = model(x)
            y_val_NIPTS_3000_true += list(y1.squeeze().cpu().numpy())
            y_val_NIPTS_4000_true += list(y2.squeeze().cpu().numpy())
            y_val_NIPTS_6000_true += list(y3.squeeze().cpu().numpy())
            y_val_NIPTS_3000_predict += list(
                predict[0].squeeze().cpu().detach().numpy())
            y_val_NIPTS_4000_predict += list(
                predict[1].squeeze().cpu().detach().numpy())
            y_val_NIPTS_6000_predict += list(
                predict[2].squeeze().cpu().detach().numpy())
            loss_1 = loss_function(predict[0], y1.unsqueeze(1).float())
            loss_2 = loss_function(predict[1], y2.unsqueeze(1).float())
            loss_3 = loss_function(predict[2], y3.unsqueeze(1).float())
            loss = loss_1 + loss_2 + loss_3
            total_eval_loss += float(loss)
            count_eval += 1
        NIPTS_3000_RMSE = root_mean_squared_error(y_val_NIPTS_3000_true,
                                                  y_val_NIPTS_3000_predict)
        NIPTS_4000_RMSE = root_mean_squared_error(y_val_NIPTS_4000_true,
                                                  y_val_NIPTS_4000_predict)
        NIPTS_6000_RMSE = root_mean_squared_error(y_val_NIPTS_6000_true,
                                                  y_val_NIPTS_6000_predict)
        logger.info(f"Epoch {i} val loss is {round(total_eval_loss/count, 2)}")
        logger.info(
            f"          NIPTS_3000's RMSE is {round(NIPTS_3000_RMSE, 2)}")
        logger.info(
            f"          NIPTS_4000's RMSE is {round(NIPTS_4000_RMSE, 2)}")
        logger.info(
            f"          NIPTS_6000's RMSE is {round(NIPTS_6000_RMSE, 2)}")

        # earl stopping
        if i == 0:
            eval_loss = total_eval_loss / count_eval
        else:
            if total_eval_loss / count_eval < eval_loss:
                eval_loss = total_eval_loss / count_eval
            else:
                if patience < early_stop:
                    patience += 1
                else:
                    logger.info(
                        f"val loss is not decrease in {patience} epoch and break training"
                    )
                    break


def data_preparation(
        train_df: pd.DataFrame,
        user_feature_col: list = ["age", "sex", "duration", "work_position"],
        label_col: list = ["NIPTS_3000", "NIPTS_4000", "NIPTS_6000"],
        categorical_col: list = ["sex", "work_position"],
        test_size: float = 0.2,
        random_state: int = 42):
    # feature engine
    for col in seq(train_df.columns).filter(lambda x: not x in label_col):
        if col in categorical_col:
            le = LabelEncoder()
            train_df.loc[:, col] = le.fit_transform(train_df[col])
        else:
            mm = MinMaxScaler()
            train_df.loc[:,
                         col] = mm.fit_transform(train_df[[col
                                                           ]]).reshape(-1)

    # user feature, item feature
    user_feature_dict, item_feature_dict = dict(), dict()
    for idx, col in enumerate(
            seq(train_df.columns).filter(lambda x: not x in label_col)):
        if col in user_feature_col:
            if col in categorical_col:
                user_feature_dict[col] = (len(train_df[col].unique()) + 1, idx)
            else:
                user_feature_dict[col] = (1, idx)
        else:
            if col in categorical_col:
                item_feature_dict[col] = (len(train_df[col].unique()) + 1, idx)
            else:
                item_feature_dict[col] = (1, idx)

    # val data
    train_data, val_data = train_test_split(train_df,
                                            test_size=test_size,
                                            random_state=random_state)
    return train_data, val_data, user_feature_dict, item_feature_dict


if __name__ == "__main__":
    from datetime import datetime
    logger.add(
        f"./log/MMoE_regression-{datetime.now().strftime('%Y-%m-%d')}.log",
        level="INFO")

    import argparse
    parser = argparse.ArgumentParser()
    # parser.add_argument("--task", type=str, default="extract")
    # parser.add_argument("--input_path",
    #                     type=str,
    #                     default="./cache/extract_data.pkl")
    parser.add_argument("--task", type=str, default="analysis")
    parser.add_argument("--input_path",
                        type=str,
                        default="./results/extract_df_MMoE.csv")
    parser.add_argument("--output_path", type=str, default="./results")
    parser.add_argument("--picture_path", type=str, default="./pictures")
    parser.add_argument("--model_path", type=str, default="./models")
    parser.add_argument("--additional_set",
                        type=dict,
                        default={
                            "mean_key": [3000, 4000, 6000],
                            "better_ear_strategy": "optimum",
                            "NIPTS_diagnose_strategy": "better"
                        })
    parser.add_argument("--n_jobs", type=int, default=-1)
    parser.add_argument("--is_show", type=bool, default=True)
    args = parser.parse_args()

    logger.info("Input Parameters informations:")
    message = "\n".join([f"{k:<20}: {v}" for k, v in vars(args).items()])
    logger.info(message)

    task = args.task
    input_path = Path(args.input_path)
    output_path = Path(args.output_path)
    picture_path = Path(args.picture_path)
    model_path = Path(args.model_path)
    additional_set = args.additional_set
    n_jobs = args.n_jobs
    is_show = args.is_show

    for output in (output_path, picture_path, model_path):
        if not output.exists():
            output.mkdir(parents=True)

    if task == "extract":
        original_data = pickle.load(open(input_path, "rb"))

        extract_df = extract_data_for_task(df=original_data,
                                           n_jobs=n_jobs,
                                           **additional_set)
        extract_df.index = extract_df.staff_id
        extract_df.drop("staff_id", axis=1, inplace=True)
        extract_df.to_csv(output_path / "extract_df_MMoE.csv",
                          header=True,
                          index=True)

    if task == "analysis":
        extract_df = pd.read_csv(input_path, header=0, index_col="staff_id")
        logger.info(f"Size: {extract_df.shape}")
        extract_df.dropna(how="any", axis=0, inplace=True)
        logger.info(f"Size after dropna: {extract_df.shape}")

        train_data, val_data, user_feature_dict, item_feature_dict = data_preparation(
            train_df=extract_df,
            user_feature_col=["age", "sex", "duration", "work_position"],
            label_col=["NIPTS_3000", "NIPTS_4000", "NIPTS_6000", "NIPTS"],
            categorical_col=["sex", "work_position"],
            test_size=0.2,
            random_state=42)
        feature_columns = seq(
            extract_df.columns).filter(lambda x: not x.startswith("NIPTS"))
        train_dataset = (train_data[feature_columns].values,
                         train_data["NIPTS_3000"].values,
                         train_data["NIPTS_4000"].values,
                         train_data["NIPTS_6000"].values)
        val_dataset = (val_data[feature_columns].values,
                       val_data["NIPTS_3000"].values,
                       val_data["NIPTS_4000"].values,
                       val_data["NIPTS_6000"].values)
        train_dataset = TrainDataSet(train_dataset)
        val_dataset = TrainDataSet(val_dataset)

        # dataloader
        train_dataloader = DataLoader(train_dataset,
                                      batch_size=64,
                                      shuffle=True)
        val_dataloader = DataLoader(val_dataset, batch_size=64, shuffle=False)

        # pytorch优化参数
        learn_rate = 0.01
        mse_loss = nn.MSELoss()
        early_stop = 3

        # train model
        # mmoe
        mmoe = MMoE(user_feature_dict,
                    item_feature_dict,
                    emb_dim=64,
                    n_expert=3,
                    num_task=3)
        optimizer = torch.optim.Adam(mmoe.parameters(), lr=learn_rate)
        train_model(model=mmoe,
                    train_loader=train_dataloader,
                    val_loader=val_dataloader,
                    epoch=2000,
                    loss_function=mse_loss,
                    optimizer=optimizer,
                    path="./models/model_mmoe_{}",
                    early_stop=early_stop)
        
        predict_res = []
        for batch in val_dataloader:
            x, y1, y2, y3 = batch
            y1_pred, y2_pred, y3_pred = mmoe(x)
            y1_pred = y1_pred.squeeze(1).cpu().detach().numpy()
            y2_pred = y2_pred.squeeze(1).cpu().detach().numpy()
            y3_pred = y3_pred.squeeze(1).cpu().detach().numpy()
            y_total = np.mean((y1_pred, y2_pred, y3_pred), axis=0)

    print(1)
