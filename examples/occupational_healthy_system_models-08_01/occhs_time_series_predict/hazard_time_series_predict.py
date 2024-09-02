# -*- coding: utf-8 -*-
"""
@DATE: 2024-04-10 17:32:43
@Author: Liu Hengjiang
@File: examples\\time_series_predict\district_time_series_predict.py
@Software: vscode
@Description:
        根据汇总的数据，分不同地级市进行每月进行体检并确诊的人数预测
"""

import re
import ast
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.dates import AutoDateLocator, AutoDateFormatter
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from prophet import Prophet
from pathlib import Path
from functional import seq
from loguru import logger

from utils.data_helper import timeseries_train_test_split

from matplotlib.font_manager import FontProperties
from matplotlib import rcParams

config = {
    "font.family": "serif",
    "font.size": 12,
    "mathtext.fontset":
    "stix",  # matplotlib渲染数学字体时使用的字体，和Times New Roman差别不大
    "font.serif": ["STZhongsong"],  # 华文中宋
    "axes.unicode_minus": False  # 处理负号，即-号
}
rcParams.update(config)
    

CITY_NAMES = {
    "杭州市": "Hangzhou",
    "丽水市": "Lishui",
    "温州市": "Wenzhou",
    "绍兴市": "Shaoxing",
    "舟山市": "Zhoushan",
    "宁波市": "Ningbo",
    "衢州市": "Quzhou",
    "嘉兴市": "Jiaxing",
    "金华市": "Jinhua",
    "台州市": "Taizhou",
    "湖州市": "Huzhou"
}

HAZARD_TYPE_NAMES = {
    "高温(高温作业)": "high_temperature",
    "正己烷": "n_hexane",
    "甲醇": "methyl_alcohol",
    "噪声": "noise",
}

DIAGNOISE_TYPE_NAMES = {
    "职业禁忌证": "occupational_contraindication",
    "疑似职业病": "suspected_occupational_disease",
    "复查": "reexamination"
}

DIAGNOISE_TYPE_DICTS = {
    "上岗前职业健康检查": ["职业禁忌证"],
    "在岗期间职业健康检查": ["职业禁忌证", "疑似职业病", "复查"],
    "离岗时职业健康检查": ["疑似职业病", "复查"],
    "离岗后健康检查": ["疑似职业病", "复查"],
    "应急健康检查": ["疑似职业病", "复查"]
}


def prophet_model(train_X, train_y, **kwargs):
    changepoint_prior_scale = kwargs.get("changepoint_prior_scale", 0.05)
    daily_seasonality = kwargs.get("daily_seasonality", False)
    yearly_seasonality = kwargs.get("yearly_seasonality", False)
    weekly_seasonality = kwargs.get("weekly_seasonality", True)

    df = pd.DataFrame({
        'ds': train_X.values,
        'y': train_y.values,
    })
    m = Prophet(
        changepoint_prior_scale=changepoint_prior_scale,
        daily_seasonality=daily_seasonality,
        yearly_seasonality=yearly_seasonality,  #年周期性
        weekly_seasonality=weekly_seasonality,  #周周期性
        #         growth="logistic",
    )
    m.add_seasonality(name='monthly',
                      period=30.5,
                      fourier_order=5,
                      prior_scale=0.1)  #月周期性
    m.add_country_holidays(country_name='CN')  #中国所有的节假日
    m.fit(df)
    return m


def plot_forecast_res(models_path,
                      output_path,
                      pictures_path,
                      city: str = "杭州市",
                      hazard_types: list = [],
                      future_periods: int = 180,
                      xlabel="日期",
                      ylabel="危害因素暴露人数",
                      **kwargs):
    
    
    model_pred_dict = {}
    for col in hazard_types:
        data = pd.read_csv(output_path / 
                       f"{CITY_NAMES.get(city)}-hazard_{HAZARD_TYPE_NAMES.get(col)}-data.csv",
                       parse_dates=["report_issue_date"],
                       header=0, index_col="report_issue_date")
        try:
            data_train = data.query("data_type == 'train'").resample("W").sum()
            data_test = data.query("data_type == 'test'").resample("W").sum()
        except:
            raise ValueError("数据量太少，无法进行建模")

        model = pickle.load(
        open(
            models_path /
            f"{CITY_NAMES.get(city)}-hazard_{HAZARD_TYPE_NAMES.get(col)}-model.pkl",
            "rb"))
        future = model.make_future_dataframe(
            periods=future_periods,
            freq='D')  #预测时长
        fcst = model.predict(future)
        fcst.index = fcst['ds'].dt.to_pydatetime()
        
        model_pred_dict[col] = {"train": data_train, "test": data_test, "predict": fcst.resample("W").sum()}
    
    fig, ax = plt.subplots(1, figsize=(15, 5))
    for col in hazard_types:
        data_train = model_pred_dict[col]["train"]
        data_test = model_pred_dict[col]["test"]
        fcst_base = model_pred_dict[col]["predict"]
        train_scatter = ax.scatter(data_train.index, data_train["exam_nums"], marker="x")
        test_scatter = ax.scatter(data_test.index, data_test["exam_nums"], marker="+", color=train_scatter.get_facecolor())
        ax.plot(fcst_base.index, fcst_base["yhat"], ls="-", label=f"{col}-危害因素暴露人数")
    
    locator = AutoDateLocator(interval_multiples=False)
    formatter = AutoDateFormatter(locator)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)
    ax.grid(True, which="major", c="gray", ls="-", lw=1, alpha=0.2)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(f"{city}-部分危害因素暴露情况及未来趋势")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(pictures_path /
                    f"{CITY_NAMES.get(city)}-hazard_exposure-res.png")
    plt.close(fig=fig)
    return fig


def train_step(input_path, models_path, output_path):

    hazard_input_df = pd.read_csv(input_path /
                                     "hazard_time_series_data.csv",
                                     header=0)
    hazard_input_df["report_issue_date"] = pd.to_datetime(
        hazard_input_df["report_issue_date"])
    hazard_input_df.sort_values(by="report_issue_date",
                                   ascending=True,
                                   inplace=True)

    city_list = hazard_input_df["organization_city"].drop_duplicates()
    for city in city_list:
        logger.info(f"Start to process the data in city: {city}")
        hazard_sub_city_df = hazard_input_df.query("organization_city == @city")
        hazard_type_list = hazard_sub_city_df["hazard_type"].drop_duplicates()
        for hazard_type in HAZARD_TYPE_NAMES.keys():
        # for hazard_type in hazard_type_list:
            logger.info(f"Start to process the data in exam type: {hazard_type}")
            hazard_sub_df = hazard_sub_city_df.query("hazard_type == @hazard_type")
            # 统计城市-危害因素类型下参加职业健康检查的人数
            hazard_timeseries_sub_df = hazard_sub_df.groupby(
                ["report_issue_date"]).sum()
            hazard_timeseries_sub_df.rename(
                columns={"hazard_res": "exam_nums"}, inplace=True)

            # 训练时间序列模型
            if hazard_timeseries_sub_df.shape[0] > 10:
                train_X, test_X, train_y, test_y = timeseries_train_test_split(
                    X=pd.Series(hazard_timeseries_sub_df.index),
                    y=hazard_timeseries_sub_df,
                    train_size=0.8)
                for col in train_y.columns:
                    logger.info(f"Start to analysis the {col} data")
                    model = prophet_model(train_X=train_X,
                                          train_y=train_y[col])
                    pickle.dump(
                        model,
                        open(
                            models_path /
                            f"{CITY_NAMES.get(city)}-hazard_{HAZARD_TYPE_NAMES.get(hazard_type)}-model.pkl",
                            "wb"))
                train_y["data_type"] = "train"
                test_y["data_type"] = "test"
                diagnoise_timeseries_output_df = pd.concat([train_y, test_y])
            else:
                logger.warning(f"{city}-{hazard_type}数据量少于十条")
                diagnoise_timeseries_output_df = hazard_timeseries_sub_df
            diagnoise_timeseries_output_df.to_csv(
                output_path / 
                f"{CITY_NAMES.get(city)}-hazard_{HAZARD_TYPE_NAMES.get(hazard_type)}-data.csv",
                header=True, index=True)
                


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, default="./cache")
    parser.add_argument("--output_path", type=str, default="./results")
    parser.add_argument("--pictures_path", type=str, default="./pictures")
    parser.add_argument("--models_path", type=str, default="./models")
    # parser.add_argument("--task", type=str, default="train")
    parser.add_argument("--task", type=str, default="plot")
    args = parser.parse_args()

    logger.info("Input Parameters informations:")
    message = "\n".join([f"{k:<20}: {v}" for k, v in vars(args).items()])
    logger.info(message)

    input_path = Path(args.input_path)
    output_path = Path(args.output_path)
    pictures_path = Path(args.pictures_path)
    models_path = Path(args.models_path)
    task = args.task

    for path in (output_path, pictures_path, models_path):
        if not path.exists():
            path.mkdir(parents=True)

    if task == "train":
        train_step(input_path, models_path, output_path)
    if task == "plot":
        city = "杭州市"
        hazard_types = ["正己烷", "噪声", "甲醇", "高温(高温作业)"]
        plot_forecast_res(models_path=models_path,
                          output_path=output_path,
                          pictures_path=pictures_path,
                          city=city,
                          hazard_types=hazard_types)
    print(1)
