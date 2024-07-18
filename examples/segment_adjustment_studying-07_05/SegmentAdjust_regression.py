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
import random
import numpy as np
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


# 设置随机种子
def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    

class SegmentAdjsttStepRunner(StepRunner):

    def step(self, features, label):
        # loss
        pred = self.model(features)
        loss = self.loss_fn(pred, label.unsqueeze(1).float())
        # 对网络参数进行非负裁剪以及添加L2权重正则化项
        l2_regularization = torch.tensor(0.)
        for param in self.model.parameters():
            param.data.clamp_(min=1e-3)
            l2_regularization += torch.norm(param, 2)

        weight_decay = 0.01
        loss += weight_decay * l2_regularization

        # backward()
        if self.optimizer is not None and self.stage == "train":
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                           max_norm=1.0)
            # 使用钩子函数强制修改NAN的参数为零
            with torch.no_grad():
                for param in self.model.parameters():
                    if torch.isnan(param).any():
                        param[torch.isnan(param)] = 0

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
    # parser.add_argument("--task", type=str, default="test")
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
    seed = 2024
    set_random_seed(seed)  # 设置随机种子
    train_dataloader = DataLoader(train_dataset,
                                  batch_size=512,
                                  shuffle=True,
                                  generator=torch.Generator().manual_seed(seed))

    # train model
    segment_adjust = SegmentAdjustModel(in_features=480, out_features=1)

    if task == "train":
        # pytorch优化参数
        set_random_seed(seed)  # 设置随机种子
        learn_rate = 0.001
        weight_decay = 0.01
        mse_loss = nn.MSELoss()
        optimizer = torch.optim.Adam(segment_adjust.parameters(),
                                     lr=learn_rate,
                                     weight_decay=weight_decay)
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
        hat_lambda = segment_adjust.state_dict(
        )["custom_layer.hat_lambda"].cpu().numpy()[0]
        logger.info(f"weight_decay: {weight_decay}, \
                      hat_lambda: {hat_lambda}")

        # weight_0 = segment_adjust.state_dict()["custom_layer.weight_0"].cpu(
        # ).numpy()[0]
        # weight_1 = segment_adjust.state_dict()["custom_layer.weight_1"].cpu(
        # ).numpy()[0]
        # linear_weight = segment_adjust.state_dict()["linear.weight"].cpu(
        # ).numpy()[0]
        # linear_bias = segment_adjust.state_dict()["linear.bias"].cpu().numpy(
        # )[0]
        # logger.info(f"weight_decay: {weight_decay}, \
        #             hat_lambda_1: {weight_1.mean()/weight_0.mean()},  \
        #             hat_lambda_2: {(weight_1/weight_0).mean()}")

    else:
        segment_adjust.load_state_dict(
            torch.load(model_path / "SegmentAdjust_checkpoint.pt"))
        hat_lambda = segment_adjust.state_dict(
        )["custom_layer.hat_lambda"].cpu().numpy()[0]
        logger.info(f"hat_lambda: {hat_lambda}")
        # weight_0 = segment_adjust.state_dict()["custom_layer.weight_0"].cpu(
        # ).numpy()[0]
        # weight_1 = segment_adjust.state_dict()["custom_layer.weight_1"].cpu(
        # ).numpy()[0]
        # linear_weight = segment_adjust.state_dict()["linear.weight"].cpu(
        # ).numpy()[0]
        # linear_bias = segment_adjust.state_dict()["linear.bias"].cpu().numpy(
        # )[0]

    print(1)
