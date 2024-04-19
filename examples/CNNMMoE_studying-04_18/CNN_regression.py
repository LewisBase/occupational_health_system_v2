# -*- coding: utf-8 -*-
"""
@DATE: 2024-04-18 11:39:40
@Author: Liu Hengjiang
@File: examples\CNNMMoE_studying-04_18\CNN_regression.py
@Software: vscode
@Description:
        进行CNN模型回归测试
"""

import pickle
from pathlib import Path
from loguru import logger
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from model.conv_layer.cnn import CNNModel
from utils.data_helper import root_mean_squared_error
from extract_Chinese_tensor_data import TrainDataSet


def train_model(model: nn.Module,
                train_loader: DataLoader,
                val_loader: DataLoader,
                epoch: int,
                loss_function,
                optimizer,
                path: str,
                early_stop: int,
                checkpoint: int = 10):
    """_summary_

    Args:
        model (nn.Module): _description_
        train_loader (DataLoader): _description_
        val_loader (DataLoader): _description_
        epoch (int): _description_
        loss_function (_type_): _description_
        optimizer (_type_): _description_
        path (Path): _description_
        early_stop (int): _description_
    """
    # use GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # 多少步内验证集的loss没有变小就提前停止
    patience, eval_loss = 0, 0

    # train
    for i in range(epoch):
        y_train_NIPTS_346_true = []
        y_train_NIPTS_346_predict = []
        total_loss, count = 0, 0
        for idx, (x, y, _, _, _, _) in tqdm(enumerate(train_loader),
                                         total=len(train_loader)):
            x, y = x.to(device), y.to(device)
            x = x.unsqueeze(1)
            # logger.info(f"x shape after unsqueeze: {x.shape}")
            predict = model(x)
            y_train_NIPTS_346_true += list(y.squeeze().cpu().numpy())
            y_train_NIPTS_346_predict += list(
                predict.squeeze().cpu().detach().numpy())
            loss = loss_function(predict, y.unsqueeze(1).float())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += float(loss)
            count += 1
        if (i + 1) % checkpoint == 0:
            torch.save(model, path.format(i + 1))
        NIPTS_346_RMSE = root_mean_squared_error(y_train_NIPTS_346_true,
                                                 y_train_NIPTS_346_predict)
        logger.info(f"Epoch {i} train loss is {round(total_loss/count, 2)}")
        logger.info(
            f"          NIPTS_346's RMSE is {round(NIPTS_346_RMSE, 2)}")

        # 验证
        total_eval_loss = 0
        model.eval()
        count_eval = 0
        y_val_NIPTS_346_true = []
        y_val_NIPTS_346_predict = []
        for idx, (x, y, _, _, _, _) in tqdm(enumerate(val_loader),
                                         total=len(val_loader)):
            x, y = x.to(device), y.to(device)
            x = x.unsqueeze(1)
            predict = model(x)
            y_val_NIPTS_346_true += list(y.squeeze().cpu().numpy())
            y_val_NIPTS_346_predict += list(
                predict.squeeze().cpu().detach().numpy())
            loss = loss_function(predict, y.unsqueeze(1).float())
            total_eval_loss += float(loss)
            count_eval += 1
        NIPTS_346_RMSE = root_mean_squared_error(y_val_NIPTS_346_true,
                                                 y_val_NIPTS_346_predict)
        logger.info(f"Epoch {i} val loss is {round(total_eval_loss/count, 2)}")
        logger.info(
            f"          NIPTS_346's RMSE is {round(NIPTS_346_RMSE, 2)}")

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
    torch.save(model, path.format(i + 1))


if __name__ == "__main__":
    from datetime import datetime
    logger.add(
        f"./log/CNN_regression-{datetime.now().strftime('%Y-%m-%d')}.log",
        level="INFO")

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_path",
        type=str,
        default="./results")
    parser.add_argument("--model_path", type=str, default="./models")
    args = parser.parse_args()

    logger.info("Input Parameters informations:")
    message = "\n".join([f"{k:<20}: {v}" for k, v in vars(args).items()])
    logger.info(message)

    input_path = Path(args.input_path)
    model_path = Path(args.model_path)

    # for output in (model_path):
    #     if not output.exists():
    #         output.mkdir(parents=True)

    ## load tensor data from pkl
    train_dataset = pickle.load(open(input_path / "train_dataset.pkl", "rb"))
    val_dataset = pickle.load(open(input_path / "val_dataset.pkl", "rb"))

    # dataloader
    train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=64, shuffle=False)

    # pytorch优化参数
    learn_rate = 0.01
    mse_loss = nn.MSELoss()
    early_stop = 100

    # train model
    # mmoe
    cnn = CNNModel()
    optimizer = torch.optim.Adam(cnn.parameters(), lr=learn_rate)
    train_model(model=cnn,
                train_loader=train_dataloader,
                val_loader=val_dataloader,
                epoch=2000,
                loss_function=mse_loss,
                optimizer=optimizer,
                path="./models/model_cnn_{}",
                early_stop=early_stop,
                checkpoint=50)

    standard_res = []
    ISO_predict_res = []
    for batch in val_dataloader:
        _, y, _, _, _, ISO_predict = batch
        standard_res += list(y.squeeze().cpu().numpy())
        ISO_predict_res += list(ISO_predict.squeeze().cpu().numpy())
    ISO_RMSE = root_mean_squared_error(standard_res, ISO_predict_res)
    logger.info(f"ISO predict NIPTS_346's RMSE is {round(ISO_RMSE, 2)}")

    print(1)
