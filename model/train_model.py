import sys
from pickletools import optimize
from loguru import logger
from pydantic import BaseModel
from typing import Any, Dict
import numpy as np
import pandas as pd
import datetime
from tqdm import tqdm
import torch
from torch import nn
from copy import deepcopy


def printlog(info):
    nowtime = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    logger.info("\n"+"=========="*8+"%s" % nowtime)
    logger.info(str(info)+"\n")


class StepRunner(BaseModel):
    model: Any 
    loss_fn: Any
    metrics_dict: Dict
    stage: str = "train"
    optimizer: Any = None

    def step(self, features, labels):
        label = labels
        # loss
        preds = self.model(features).squeeze(-1)
        loss = self.loss_fn(preds, label)

        # backward()
        if self.optimizer is not None and self.stage == "train":
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

        # metrics
        step_metrics = {self.stage+"_"+name: metric_fn(preds, label.int()).item()
                        for name, metric_fn in self.metrics_dict.items()}
        return loss.item(), step_metrics

    def train_step(self, features, labels):
        self.model.train()  # 训练模式，dropout层发生作用
        return self.step(features, labels)

    @torch.no_grad()
    def eval_step(self, features, labels):
        self.model.eval()  # 预测模式，dropout层不发生作用
        return self.step(features, labels)

    def __call__(self, features, labels):
        if self.stage == "train":
            return self.train_step(features, labels)
        else:
            return self.eval_step(features, labels)


class EpochRunner():
    def __init__(self, steprunner: StepRunner):
        self.steprunner = steprunner
        self.stage = self.steprunner.stage

    def __call__(self, dataloader, device):
        total_loss, step = 0, 0
        loop = tqdm(enumerate(dataloader), total=len(dataloader))
        for i, batch in loop:
            loss, step_metrics = self.steprunner(*batch)
            step_log = dict({self.stage+"_loss": loss}, **step_metrics)
            total_loss += loss
            step += 1
            if i != len(dataloader)-1:
                loop.set_postfix(**step_log)
            else:
                epoch_loss = total_loss/step
                epoch_metrics = {self.stage+"_"+name: metric_fn.compute().item()
                                 for name, metric_fn in self.steprunner.metrics_dict.items()}
                epoch_log = dict(
                    {self.stage+"_loss": epoch_loss}, **epoch_metrics)
                loop.set_postfix(**epoch_log)

                for name, metric_fn in self.steprunner.metrics_dict.items():
                    metric_fn.reset()
        return epoch_log


def train_model(model,
                steprunner,
                epochrunner,
                optimizer,
                loss_fn,
                metrics_dict,
                train_data,
                val_data=None,
                epochs=10,
                ckpt_path="checkpoint.pt",
                early_stop=5,
                monitor="val_loss",
                mode="min"):
    history = {}
    # use GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    for epoch in range(1, epochs + 1):
        printlog(f"Epoch {epoch} / {epochs}")

        # 1, train -------------------------------------------------
        train_step_runner = steprunner(model=model,
                                       stage="train",
                                       loss_fn=loss_fn,
                                       metrics_dict=deepcopy(metrics_dict),
                                       optimizer=optimizer)
        train_epoch_runner = epochrunner(train_step_runner)
        train_metrics = train_epoch_runner(train_data, device=device)

        for name, metric in train_metrics.items():
            history[name] = history.get(name, []) + [metric]

        # 2，validate -------------------------------------------------
        if val_data:
            val_step_runner = steprunner(model=model,
                                         stage="val",
                                         loss_fn=loss_fn,
                                         metrics_dict=deepcopy(metrics_dict))
            val_epoch_runner = epochrunner(val_step_runner)
            with torch.no_grad():
                val_metrics = val_epoch_runner(val_data, device=device)
            val_metrics["epoch"] = epoch
            for name, metric in val_metrics.items():
                history[name] = history.get(name, []) + [metric]

        # 3，early-stopping -------------------------------------------------
        arr_scores = history[monitor]
        best_score_idx = np.argmax(arr_scores) if mode == "max" else np.argmin(
            arr_scores)
        if best_score_idx == len(arr_scores) - 1:
            torch.save(model.state_dict(), ckpt_path)
            logger.info(
                f"<<<<<< reach best {monitor} : {arr_scores[best_score_idx]} >>>>>>"
            )
        if len(arr_scores) - best_score_idx > early_stop:
            logger.info(
                f"<<<<<< {monitor} without improvement in {early_stop} epoch, early stopping >>>>>>"
            )
            break
        model.load_state_dict(torch.load(ckpt_path))

    return pd.DataFrame(history)


if __name__ == "__main__":
    from torchmetrics import Accuracy
    from catboost.datasets import titanic
    from sklearn.model_selection import train_test_split
    from torch.utils.data import DataLoader, TensorDataset
    
    train_df, test_df = titanic()

    def preprocessing(dfdata):
        dfresult = pd.DataFrame()
    
        # Pclass
        dfPclass = pd.get_dummies(dfdata['Pclass'])
        dfPclass.columns = ["Pclass_"+str(x) for x in dfPclass.columns]
        dfresult = pd.concat([dfresult, dfPclass], axis = 1)
    
        # Sex
        dfSex = pd.get_dummies(dfdata["Sex"])
        dfresult = pd.concat([dfresult,dfSex],axis=1)
    
        # Age
        dfresult["Age"] = dfdata["Age"].fillna(0)
        dfresult["Age_null"] = pd.isna(dfdata["Age"]).astype("int32")
    
        # SibSp, Parch, Fare
        dfresult["SibSP"] = dfdata["SibSp"]
        dfresult["Parch"] = dfdata["Parch"]
        dfresult["Fare"] = dfdata["Fare"]
    
        # Carbin
        dfresult["Cabin_null"] = pd.isna(dfdata["Cabin"]).astype("int32")
    
        # Embarked
        dfEmbarked = pd.get_dummies(dfdata["Embarked"], dummy_na=True)
        dfEmbarked.columns = ["Embarked_"+str(x) for x in dfEmbarked.columns]
        dfresult = pd.concat([dfresult,dfEmbarked],axis=1)
    
        return dfresult

    X = preprocessing(train_df).values
    y = train_df["Survived"].values
    
    x_train,x_val,y_train,y_val = train_test_split(X,y,train_size=0.8)
    x_test = preprocessing(test_df).values

    dl_train = DataLoader(TensorDataset(torch.tensor(x_train).float(
    ), torch.tensor(y_train).float()), shuffle=True, batch_size=8)
    dl_val = DataLoader(TensorDataset(torch.tensor(x_val).float(
    ), torch.tensor(y_val).float()), shuffle=True, batch_size=8)

    def create_net():
        net = nn.Sequential()
        net.add_module("linear1", nn.Linear(15,20))
        net.add_module("relu1", nn.ReLU())
        net.add_module("linear2", nn.Linear(20,15))
        net.add_module("relu2", nn.ReLU())
        net.add_module("linear3", nn.Linear(15,1))
        net.add_module("sigmoid", nn.Sigmoid())
        return net

    net = create_net()
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.01)
    metrics_dict = {"acc": Accuracy(task="BINARY")}

    dfhistory = train_model(net,
                            StepRunner,
                            EpochRunner,
                            optimizer,
                            loss_fn,
                            metrics_dict,
                            train_data=dl_train,
                            val_data=dl_val,
                            epochs=10,
                            early_stop=5,
                            monitor="val_acc",
                            mode="max")
    print(1)