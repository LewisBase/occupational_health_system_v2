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

CITY_NAMES= {
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


def prophet_model(train_X, train_y, **kwargs):
    changepoint_prior_scale = kwargs.get("changepoint_prior_scale", 0.05)
    daily_seasonality = kwargs.get("daily_seasonality", False)
    yearly_seasonality = kwargs.get("yearly_seasonality", False)
    weekly_seasonality = kwargs.get("weekly_seasonality", True)

    df = pd.DataFrame({
        'ds': train_X,
        'y': train_y,
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


def plot_forecast_res(model,
                      fcst,
                      test_X,
                      test_y,
                      title,
                      xlabel='date',
                      ylabel='y',
                      uncertainty=True,
                      plot_cap=True,
                      **kwargs):
    fig, ax = plt.subplots(1, figsize=(10, 4))
    fcst_t = fcst['ds'].dt.to_pydatetime()
    forecast_y = fcst[fcst["ds"].isin(test_X)]["yhat"]
    rmse = mean_squared_error(test_y, forecast_y)
    mape = mean_absolute_percentage_error(test_y, forecast_y)
    ax.plot(model.history['ds'].dt.to_pydatetime(),
            model.history['y'],
            'k.',
            label='历史观测数据（训练）')
    ax.plot(test_X.dt.to_pydatetime(), test_y, 'r.', label='历史观测数据（测试）')
    ax.plot(fcst_t, fcst['yhat'], ls='-', c='#0072B2', label='模型预测结果')
    if 'cap' in fcst and plot_cap:
        ax.plot(fcst_t, fcst['cap'], ls='--', c='k')
    if model.logistic_floor and 'floor' in fcst and plot_cap:
        ax.plot(fcst_t, fcst['floor'], ls='--', c='k')
    if uncertainty and model.uncertainty_samples:
        ax.fill_between(fcst_t,
                        fcst['yhat_lower'],
                        fcst['yhat_upper'],
                        color='#0072B2',
                        alpha=0.2)
    ax.annotate(
        text=f"RMSE={round(rmse,2)}\nMAPE={round(mape,2)}",
        xy=(0.5, 0.8),
        xycoords='axes fraction',
        xytext=(0.5, 0.8),
        textcoords='axes fraction',
    )
    locator = AutoDateLocator(interval_multiples=False)
    formatter = AutoDateFormatter(locator)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)
    ax.grid(True, which='major', c='gray', ls='-', lw=1, alpha=0.2)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    return fig


def step(input_path, pictures_path, models_path, task, plot_types):
    diagnoise_input_df = pd.read_csv(input_path /
                                     "diagnoise_time_series_data.csv",
                                     header=0)
    diagnoise_input_df["report_issue_date"] = pd.to_datetime(
        diagnoise_input_df["report_issue_date"])
    diagnoise_input_df.sort_values(by="report_issue_date",
                                   ascending=True,
                                   inplace=True)
    hazard_input_df = pd.read_csv(input_path / "hazard_time_series_data.csv",
                                  header=0)
    hazard_input_df["report_issue_date"] = pd.to_datetime(
        hazard_input_df["report_issue_date"])
    hazard_input_df.sort_values(by="report_issue_date",
                                ascending=True,
                                inplace=True)

    top10_hazard_info = {}
    for city in diagnoise_input_df["organization_city"].drop_duplicates():
        logger.info(f"Start to analysis the data in city: {city}")
        hazard_sub_df = hazard_input_df[hazard_input_df["organization_city"] ==
                                        city]
        top10_hazard_prop = hazard_sub_df.groupby("hazard_type")["hazard_num"].sum().to_frame()
        top10_hazard_prop["hazard_prop"] = top10_hazard_prop["hazard_num"] / top10_hazard_prop["hazard_num"].sum()
        top10_hazard_dict = top10_hazard_prop.sort_values(by="hazard_prop", ascending=False)["hazard_prop"].head(10).to_dict()
        top10_hazard_info[city] = top10_hazard_dict

        diagnoise_sub_df = diagnoise_input_df[
            diagnoise_input_df["organization_city"] == city]
        train_X, test_X, train_y, test_y = timeseries_train_test_split(
            X=diagnoise_sub_df["report_issue_date"],
            y=diagnoise_sub_df[["exam_num", "diagnoise_num"]],
            train_size=0.8)
        for col in train_y.columns:
            logger.info(f"Start to analysis the {col} data")
            if task == "train":
                model = prophet_model(train_X=train_X, train_y=train_y[col])
                pickle.dump(
                    model,
                    open(models_path / f"{CITY_NAMES.get(city)}-diagnoise-{col}-model.pkl",
                         "wb"))
            else:
                model = pickle.load(
                    open(models_path / f"{CITY_NAMES.get(city)}-diagnoise-{col}-model.pkl",
                         "rb"))
            future = model.make_future_dataframe(
                periods=(test_X.iloc[-1] - test_X.iloc[0]).days + 1,
                freq='D')  #预测时长
            forecast = model.predict(future)
            if "comp" in plot_types:
                fig_comp = model.plot_components(forecast)
                fig_comp.savefig(pictures_path /
                                 f"{city}-diagnoise-{col}_comp.png")
                plt.close(fig=fig_comp)
            if "res" in plot_types:
                fig_res = plot_forecast_res(model=model,
                                            fcst=forecast,
                                            test_X=test_X,
                                            test_y=test_y[col],
                                            ylabel=col,
                                            title=city)
                # fig_res.savefig(pictures_path /
                #                 f"{city}-diagnoise-{col}_res.png")
                plt.close(fig=fig_res)
    pickle.dump(top10_hazard_info, open(models_path / "top10_hazard_info.pkl", "wb"))

if __name__ == "__main__":
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

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, default="./cache")
    parser.add_argument("--output_path", type=str, default="./results")
    parser.add_argument("--pictures_path", type=str, default="./pictures")
    parser.add_argument("--models_path", type=str, default="./models")
    # parser.add_argument("--task", type=str, default="predict")
    parser.add_argument("--task", type=str, default="train")
    parser.add_argument("--plot_types", type=list, default=["comp", "res"])
    args = parser.parse_args()

    logger.info("Input Parameters informations:")
    message = "\n".join([f"{k:<20}: {v}" for k, v in vars(args).items()])
    logger.info(message)

    input_path = Path(args.input_path)
    output_path = Path(args.output_path)
    pictures_path = Path(args.pictures_path)
    models_path = Path(args.models_path)
    task = args.task
    plot_types = args.plot_types

    for path in (output_path, pictures_path, models_path):
        if not path.exists():
            path.mkdir(parents=True)

    step(input_path, pictures_path, models_path, task, plot_types)
    print(1)
