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
from torchmetrics import MeanSquaredError

from model.multi_task.cnn_mmoe import ConvMMoEModel
from model.train_model import StepRunner, EpochRunner, train_model
from extract_Chinese_tensor_data import TrainDataSet
from utils.data_helper import root_mean_squared_error


class CNNMMoEStepRunner(StepRunner):
    def step(self, features, labels):
        label, sublabel_1, sublabel_2, sublabel_3 = labels
        # loss
        preds = self.model(features)
        loss_1 = self.loss_fn(preds[0], sublabel_1.unsqueeze(1).float())
        loss_2 = self.loss_fn(preds[1], sublabel_2.unsqueeze(1).float())
        loss_3 = self.loss_fn(preds[2], sublabel_3.unsqueeze(1).float())
        loss = loss_1 + loss_2 + loss_3

        # backward()
        if self.optimizer is not None and self.stage == "train":
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

        # metrics
        total_preds = torch.mean(torch.stack(preds), dim=0)
        step_metrics = {
            self.stage + "_" + name: metric_fn(total_preds,
                                               label.unsqueeze(1)).item()
            for name, metric_fn in self.metrics_dict.items()
        }
        return loss.item(), step_metrics


class CNNMMoEEpochRunner(EpochRunner):
    def __call__(self, dataloader, device):
        total_loss, step = 0, 0
        loop = tqdm(enumerate(dataloader), total=len(dataloader))
        for i, (x, y, suby_1, suby_2, suby_3, _) in loop:
            x, y, suby_1, suby_2, suby_3 = x.to(device), y.to(
                device), suby_1.to(device), suby_2.to(device), suby_3.to(
                    device)
            x = x.unsqueeze(1)
            loss, step_metrics = self.steprunner(x,
                                                 (y, suby_1, suby_2, suby_3))
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
        f"./log/CNNMMoE_regression-{datetime.now().strftime('%Y-%m-%d')}.log",
        level="INFO")

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, default="./results")
    parser.add_argument("--model_path", type=str, default="./models")
    parser.add_argument("--task", type=str, default="train")
    # parser.add_argument("--task", type=str, default="test")
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
    val_dataset = pickle.load(open(input_path / "val_dataset.pkl", "rb"))

    # dataloader
    train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=64, shuffle=False)

    # train model
    # cnn+mmoe
    cnn_mmoe = ConvMMoEModel(conv1_in_channels=1,
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
                             linear2_out_features=64,
                             num_experts=3,
                             num_tasks=3,
                             expert_hidden_units=64,
                             gate_hidden_units=32)
    if task == "train":
        # pytorch优化参数
        learn_rate = 0.01
        mse_loss = nn.MSELoss()
        optimizer = torch.optim.Adam(cnn_mmoe.parameters(), lr=learn_rate)
        dfhistory = train_model(
            model=cnn_mmoe,
            steprunner=CNNMMoEStepRunner,
            epochrunner=CNNMMoEEpochRunner,
            optimizer=optimizer,
            loss_fn=mse_loss,
            metrics_dict={"RMSE": MeanSquaredError(squared=False)},
            train_data=train_dataloader,
            val_data=val_dataloader,
            epochs=2000,
            ckpt_path=model_path / "CNNMMoE_checkpoint.pt",
            early_stop=100,
            monitor="val_loss",
            mode="min")
    else:
        cnn_mmoe.load_state_dict(
            torch.load(model_path / "CNNMMoE_checkpoint.pt"))
        standard_res = []
        standard_3000_res = []
        standard_4000_res = []
        standard_6000_res = []
        ISO_predict_res = []
        CNNMMoE_predict_res = []
        CNNMMoE_predict_3000_res = []
        CNNMMoE_predict_4000_res = []
        CNNMMoE_predict_6000_res = []
        for batch in val_dataloader:
            x, y, suby_1, suby_2, suby_3, ISO_predict = batch
            standard_res += list(y.squeeze().cpu().numpy())
            standard_3000_res += list(suby_1.squeeze().cpu().numpy())
            standard_4000_res += list(suby_2.squeeze().cpu().numpy())
            standard_6000_res += list(suby_3.squeeze().cpu().numpy())
            CNNMMoE_predict_res += list(
                torch.mean(torch.stack(cnn_mmoe(x.unsqueeze(1))),
                           dim=0).squeeze().cpu().detach().numpy())
            CNNMMoE_predict_3000_res += list(cnn_mmoe(x.unsqueeze(1))[0].squeeze().cpu().detach().numpy())
            CNNMMoE_predict_4000_res += list(cnn_mmoe(x.unsqueeze(1))[1].squeeze().cpu().detach().numpy())
            CNNMMoE_predict_6000_res += list(cnn_mmoe(x.unsqueeze(1))[2].squeeze().cpu().detach().numpy())
            ISO_predict_res += list(ISO_predict.squeeze().cpu().numpy())
        CNNMMoE_RMSE = root_mean_squared_error(standard_res, CNNMMoE_predict_res)
        CNNMMoE_3000_RMSE = root_mean_squared_error(standard_3000_res, CNNMMoE_predict_3000_res)
        CNNMMoE_4000_RMSE = root_mean_squared_error(standard_4000_res, CNNMMoE_predict_4000_res)
        CNNMMoE_6000_RMSE = root_mean_squared_error(standard_6000_res, CNNMMoE_predict_6000_res)
        ISO_RMSE = root_mean_squared_error(standard_res, ISO_predict_res)
        logger.info(f"CNN+MMoE predict NIPTS_346's RMSE is {round(CNNMMoE_RMSE, 2)}")
        logger.info(f"ISO predict NIPTS_346's RMSE is {round(ISO_RMSE, 2)}")
        logger.info(f"CNN+MMoE predict NIPTS_3000's RMSE is {round(CNNMMoE_3000_RMSE, 2)}")
        logger.info(f"CNN+MMoE predict NIPTS_4000's RMSE is {round(CNNMMoE_4000_RMSE, 2)}")
        logger.info(f"CNN+MMoE predict NIPTS_6000's RMSE is {round(CNNMMoE_6000_RMSE, 2)}")

    print(1)
