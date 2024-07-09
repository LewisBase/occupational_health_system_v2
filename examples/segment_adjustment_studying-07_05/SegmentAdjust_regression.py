# -*- coding: utf-8 -*-
"""
@DATE: 2024-07-08 11:25:03
@Author: Liu Hengjiang
@File: examples\segment_adjustment_studying-07_05\SegmentAdjust_regression.py
@Software: vscode
@Description:
        进行SegmentAdjustment方法的回归拟合测试
"""
import pickle
from pathlib import Path
from functional import seq
from loguru import logger
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchmetrics import MeanSquaredError

from model.linear_regression.custom_linear_regression import SegmentAdjustModel
from model.train_model import StepRunner, EpochRunner, train_model
from extract_all_Chinese_data import TrainDataSet
from utils.data_helper import root_mean_squared_error


class SegmentAdjsttStepRunner(StepRunner):

    def step(self, features, label):
        # loss
        pred = self.model(features)
        loss = self.loss_fn(pred, label.unsqueeze(1).float())

        # backward()
        if self.optimizer is not None and self.stage == "train":
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

        # metric
        step_metric = {
            self.stage + "_" + name: metric_fn(pred,
                                               label.unsqueeze(1)).item()
            for name, metric_fn in self.metrics_dict.items()
        }
        return loss.item(), step_metric


class SegmentAdjustEpochRunner(EpochRunner):

    def __call__(self, dataloader, device):
        total_loss, step = 0, 0
        loop = tqdm(enumerate(dataloader), total=len(dataloader))
        for i, (x, y, _) in loop:
            x, y = x.to(device), y.to(device)
            # x = x.unsqueeze(1)
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
        f"./log/SegmentAdjust_regression-{datetime.now().strftime('%Y-%m-%d')}.log",
        level="INFO")

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, default="./results")
    parser.add_argument("--model_path", type=str, default="./models")
    parser.add_argument("--task", type=str, default="train")
    args = parser.parse_args()

    logger.info("Input Parameters informations:")
    message = "\n".join([f"{k:<20}: {v}" for k, v in vars(args).items()])
    logger.info(message)

    input_path = Path(args.input_path)
    model_path = Path(args.model_path)
    task = args.task

    # mission start
    ## load tensor data from pkl
    train_dataset = pickle.load(open(input_path / "train_dataset.pkl", "rb"))

    # dataloader
    train_dataloader = DataLoader(train_dataset, batch_size=12, shuffle=True)

    # train model
    segment_adjust = SegmentAdjustModel(in_feautres=100, out_feautres=1)

    if task == "train":
        # pytorch优化参数
        learn_rate = 0.001
        mse_loss = nn.MSELoss()
        optimizer = torch.optim.Adam(segment_adjust.parameters(),
                                     lr=learn_rate)
        dfhistory = train_model(
            model=segment_adjust,
            steprunner=SegmentAdjsttStepRunner,
            epochrunner=SegmentAdjustEpochRunner,
            optimizer=optimizer,
            loss_fn=mse_loss,
            metrics_dict={"RMSE": MeanSquaredError(squared=False)},
            train_data=train_dataloader,
            val_data=train_dataloader,
            epochs=2000,
            ckpt_path=model_path / "SegmentAdjust_checkpoint.pt",
            early_stop=100,
            monitor="val_loss",
            mode="min")
    else:
        segment_adjust.load_state_dict(
            torch.load(model_path / "SegmentAdjust_checkpoint.pt"))
    
    print(1)
