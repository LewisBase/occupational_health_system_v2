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
from torchmetrics import MeanSquaredError

from model.conv_layer.cnn import CNNModel
from model.train_model import StepRunner, EpochRunner, train_model
from utils.data_helper import root_mean_squared_error
from extract_Chinese_tensor_data import TrainDataSet


class CNNStepRunner(StepRunner):
    def step(self, features, labels):
        label = labels
        # loss
        preds = self.model(features)
        loss = self.loss_fn(preds, label.unsqueeze(1).float())

        # backward()
        if self.optimizer is not None and self.stage == "train":
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

        # metrics
        step_metrics = {
            self.stage + "_" + name: metric_fn(preds,
                                               label.unsqueeze(1)).item()
            for name, metric_fn in self.metrics_dict.items()
        }
        return loss.item(), step_metrics


class CNNEpochRunner(EpochRunner):
    def __call__(self, dataloader, device):
        total_loss, step = 0, 0
        loop = tqdm(enumerate(dataloader), total=len(dataloader))
        for i, (x, y, _, _, _, _) in loop:
            x, y = x.to(device), y.to(device)
            x = x.unsqueeze(1)
            loss, step_metrics = self.steprunner(x, y)
            step_log = dict({self.stage + "_loss": loss}, **step_metrics)
            total_loss += loss
            step += 1
            if i != len(dataloader) - 1:
                loop.set_postfix(**step_log)
            else:
                epoch_loss = total_loss / step
                epoch_metrics = {
                    self.stage + "_" + name: metric_fn.compute().item()
                    for name, metric_fn in
                    self.steprunner.metrics_dict.items()
                }
                epoch_log = dict({self.stage + "_loss": epoch_loss},
                                 **epoch_metrics)
                loop.set_postfix(**epoch_log)

                for name, metric_fn in self.steprunner.metrics_dict.items():
                    metric_fn.reset()
        return epoch_log


if __name__ == "__main__":
    from datetime import datetime
    logger.add(
        f"./log/CNN_regression-{datetime.now().strftime('%Y-%m-%d')}.log",
        level="INFO")

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, default="./results")
    parser.add_argument("--model_path", type=str, default="./models")
    # parser.add_argument("--task", type=str, default="train")
    parser.add_argument("--task", type=str, default="test")
    args = parser.parse_args()

    logger.info("Input Parameters informations:")
    message = "\n".join([f"{k:<20}: {v}" for k, v in vars(args).items()])
    logger.info(message)

    input_path = Path(args.input_path)
    model_path = Path(args.model_path)
    task = args.task

    ## load tensor data from pkl
    train_dataset = pickle.load(open(input_path / "train_dataset.pkl", "rb"))
    val_dataset = pickle.load(open(input_path / "val_dataset.pkl", "rb"))

    # dataloader
    train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=64, shuffle=False)

    # train model
    # cnn
    cnn = CNNModel(conv1_in_channels=1,
                   conv1_out_channels=32,
                   conv1_kernel_size=3,
                   pool1_kernel_size=2,
                   pool1_stride=2,
                   conv2_out_channels=64,
                   conv2_kernel_size=5,
                   pool2_kernel_size=2,
                   pool2_stride=2,
                   dropout_rate=0.1,
                   adaptive_pool_output_size=(1, 1),
                   linear1_out_features=32,
                   linear2_out_features=1)
    if task == "train":
        # pytorch优化参数
        learn_rate = 0.01
        mse_loss = nn.MSELoss()
        optimizer = torch.optim.Adam(cnn.parameters(), lr=learn_rate)
        dfhistory = train_model(
            model=cnn,
            steprunner=CNNStepRunner,
            epochrunner=CNNEpochRunner,
            optimizer=optimizer,
            loss_fn=mse_loss,
            metrics_dict={"RMSE": MeanSquaredError(squared=False)},
            train_data=train_dataloader,
            val_data=val_dataloader,
            epochs=2000,
            ckpt_path=model_path / "CNN_checkpoint.pt",
            early_stop=100,
            monitor="val_loss",
            mode="min")
    else:
        cnn.load_state_dict(torch.load(model_path / "CNN_checkpoint.pt"))
        standard_res = []
        ISO_predict_res = []
        CNN_predict_res = []
        for batch in val_dataloader:
            x, y, _, _, _, ISO_predict = batch
            standard_res += list(y.squeeze().cpu().numpy())
            CNN_predict_res += list(
                cnn(x.unsqueeze(1)).squeeze().cpu().detach().numpy())
            ISO_predict_res += list(ISO_predict.squeeze().cpu().numpy())
        CNN_RMSE = root_mean_squared_error(standard_res, CNN_predict_res)
        ISO_RMSE = root_mean_squared_error(standard_res, ISO_predict_res)
        logger.info(f"CNN predict NIPTS_346's RMSE is {round(CNN_RMSE, 2)}")
        logger.info(f"ISO predict NIPTS_346's RMSE is {round(ISO_RMSE, 2)}")

    print(1)
