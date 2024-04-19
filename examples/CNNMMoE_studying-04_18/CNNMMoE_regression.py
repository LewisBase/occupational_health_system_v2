# -*- coding: utf-8 -*-
"""
@DATE: 2024-04-19 09:59:11
@Author: Liu Hengjiang
@File: examples\CNNMMoE_studying-04_18\CNNMMoE_regression.py
@Software: vscode
@Description:
        进行CNN+MMoE模型回归测试
"""
import pickle
from pathlib import Path
from loguru import logger
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from model.multi_task.cnn_mmoe import ConvMMoEModel
from extract_Chinese_tensor_data import TrainDataSet
from utils.data_helper import root_mean_squared_error


def train_model(model: nn.Module, train_loader: DataLoader,
                val_loader: DataLoader, epoch: int, loss_function, optimizer,
                path: str, early_stop: int, check_point: int = 10):
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
        y_train_NIPTS_3000_true = []
        y_train_NIPTS_3000_predict = []
        y_train_NIPTS_4000_true = []
        y_train_NIPTS_4000_predict = []
        y_train_NIPTS_6000_true = []
        y_train_NIPTS_6000_predict = []
        total_loss, count = 0, 0
        for idx, (x, y, suby_1, suby_2,
                  suby_3, _) in tqdm(enumerate(train_loader),
                                  total=len(train_loader)):
            x, y, suby_1, suby_2, suby_3 = x.to(device), y.to(device), suby_1.to(device), suby_2.to(
                device), suby_3.to(device)
            x = x.unsqueeze(1)
            # logger.info(f"x shape after unsqueeze: {x.shape}")
            predict = model(x)
            y_train_NIPTS_346_true += list(y.squeeze().cpu().numpy())
            y_train_NIPTS_346_predict += list(
                torch.mean(torch.stack(predict),dim=0).squeeze().cpu().detach().numpy())
            y_train_NIPTS_3000_true += list(suby_1.squeeze().cpu().numpy())
            y_train_NIPTS_3000_predict += list(
                predict[0].squeeze().cpu().detach().numpy())
            y_train_NIPTS_4000_true += list(suby_2.squeeze().cpu().numpy())
            y_train_NIPTS_4000_predict += list(
                predict[1].squeeze().cpu().detach().numpy())
            y_train_NIPTS_6000_true += list(suby_3.squeeze().cpu().numpy())
            y_train_NIPTS_6000_predict += list(
                predict[2].squeeze().cpu().detach().numpy())
            loss_1 = loss_function(predict[0], suby_1.unsqueeze(1).float())
            loss_2 = loss_function(predict[1], suby_2.unsqueeze(1).float())
            loss_3 = loss_function(predict[2], suby_3.unsqueeze(1).float())
            loss = loss_1 + loss_2 + loss_3
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += float(loss)
            count += 1
        if (i+1) % check_point == 0:
            torch.save(model, path.format(i + 1))
        NIPTS_346_RMSE = root_mean_squared_error(y_train_NIPTS_346_true,
                                                 y_train_NIPTS_346_predict)
        NIPTS_3000_RMSE = root_mean_squared_error(y_train_NIPTS_3000_true,
                                                  y_train_NIPTS_3000_predict)
        NIPTS_4000_RMSE = root_mean_squared_error(y_train_NIPTS_4000_true,
                                                  y_train_NIPTS_4000_predict)
        NIPTS_6000_RMSE = root_mean_squared_error(y_train_NIPTS_6000_true,
                                                  y_train_NIPTS_6000_predict)
        logger.info(f"Epoch {i} train loss is {round(total_loss/count, 2)}")
        logger.info(
            f"          NIPTS_346's RMSE is {round(NIPTS_346_RMSE, 2)}")
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
        y_val_NIPTS_346_true = []
        y_val_NIPTS_346_predict = []
        y_val_NIPTS_3000_true = []
        y_val_NIPTS_3000_predict = []
        y_val_NIPTS_4000_true = []
        y_val_NIPTS_4000_predict = []
        y_val_NIPTS_6000_true = []
        y_val_NIPTS_6000_predict = []
        for idx, (x, y, suby_1, suby_2,
                  suby_3, _) in tqdm(enumerate(val_loader), total=len(val_loader)):
            x, y, suby_1, suby_2, suby_3 = x.to(device), y.to(device), suby_1.to(device), suby_2.to(
                device), suby_3.to(device)
            x = x.unsqueeze(1)
            predict = model(x)
            y_val_NIPTS_346_true += list(y.squeeze().cpu().numpy())
            y_val_NIPTS_346_predict += list(
                torch.mean(torch.stack(predict),dim=0).squeeze().cpu().detach().numpy())
            y_val_NIPTS_3000_true += list(suby_1.squeeze().cpu().numpy())
            y_val_NIPTS_3000_predict += list(
                predict[0].squeeze().cpu().detach().numpy())
            y_val_NIPTS_4000_true += list(suby_2.squeeze().cpu().numpy())
            y_val_NIPTS_4000_predict += list(
                predict[1].squeeze().cpu().detach().numpy())
            y_val_NIPTS_6000_true += list(suby_3.squeeze().cpu().numpy())
            y_val_NIPTS_6000_predict += list(
                predict[2].squeeze().cpu().detach().numpy())
            loss_1 = loss_function(predict[0], suby_1.unsqueeze(1).float())
            loss_2 = loss_function(predict[1], suby_2.unsqueeze(1).float())
            loss_3 = loss_function(predict[2], suby_3.unsqueeze(1).float())
            loss = loss_1 + loss_2 + loss_3
            total_eval_loss += float(loss)
            count_eval += 1
        NIPTS_346_RMSE = root_mean_squared_error(y_val_NIPTS_346_true,
                                                 y_val_NIPTS_346_predict)
        NIPTS_3000_RMSE = root_mean_squared_error(y_val_NIPTS_3000_true,
                                                  y_val_NIPTS_3000_predict)
        NIPTS_4000_RMSE = root_mean_squared_error(y_val_NIPTS_4000_true,
                                                  y_val_NIPTS_4000_predict)
        NIPTS_6000_RMSE = root_mean_squared_error(y_val_NIPTS_6000_true,
                                                  y_val_NIPTS_6000_predict)
        logger.info(f"Epoch {i} val loss is {round(total_eval_loss/count, 2)}")
        logger.info(
            f"          NIPTS_346's RMSE is {round(NIPTS_346_RMSE, 2)}")
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
    torch.save(model, path.format(i + 1))


if __name__ == "__main__":
    from datetime import datetime
    logger.add(
        f"./log/CNNMMoE_regression-{datetime.now().strftime('%Y-%m-%d')}.log",
        level="INFO")

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, default="./results")
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

    # mission start
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
    cnn_mmoe = ConvMMoEModel()
    optimizer = torch.optim.Adam(cnn_mmoe.parameters(), lr=learn_rate)
    train_model(model=cnn_mmoe,
                train_loader=train_dataloader,
                val_loader=val_dataloader,
                epoch=2000,
                loss_function=mse_loss,
                optimizer=optimizer,
                path="./models/model_cnn+mmoe_{}",
                early_stop=early_stop)

    print(1)
