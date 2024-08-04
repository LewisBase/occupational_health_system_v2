# -*- coding: utf-8 -*-
"""
@DATE: 2024-05-20 15:45:33
@Author: Liu Hengjiang
@File: examples\\NOISH_1998_studying-01_26\\plot_Chinese_all_data_statistic.py
@Software: vscode
@Description:
        对中国数据中实验组与对照组进行不同维度上的统计并作图
        按照年龄与工龄分层进行针对NIHL的Mantel-Haenszel检验
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import statsmodels.api as sm
from pathlib import Path
from functional import seq
from loguru import logger

from utils.data_helper import mark_group_name
from utils.plot_helper import plot_frequency_bar


def sort_Series_by_strkey(input_Series: pd.Series, str_loc: int):
    return {
        k: v
        for k, v in sorted(input_Series.to_dict().items(),
                           key=lambda item: int(item[0][str_loc:]),
                           reverse=False)
    }


def plot_LAeq_statistic_res(experiment_df, pictures_path):
    LAeq_cut = np.arange(70, 130, 5)
    experiment_df["LAeq_box"] = experiment_df["LAeq"].apply(
        lambda x: mark_group_name(x, qcut_set=LAeq_cut, prefix="L"))
    LAeq_freq = experiment_df["LAeq_box"].value_counts(
    ) / experiment_df["LAeq_box"].value_counts().sum() * 100
    LAeq_freq = sort_Series_by_strkey(input_Series=LAeq_freq, str_loc=1)
    plot_frequency_bar(freqs={"LAeq": LAeq_freq},
                       start_point=72.5,
                       interval=5,
                       xticks=LAeq_cut,
                       xticklabels=LAeq_cut,
                       xlabel_name="$L_{Aeq,8h}$ (dBA)",
                       output_path=pictures_path,
                       picture_name="Fig2A",
                       picture_format="tiff",
                       annotations={"A": (-0.1, 1.05)})


def plot_kurtosis_statistic_res(experiment_df, pictures_path):
    kurtosis_cut = [0, 3, 10, 30, 75, np.inf]
    # experiment_df["kurtosis_arimean_box"] = experiment_df[
    #     "kurtosis_arimean"].apply(
    #         lambda x: mark_group_name(x, qcut_set=kurtosis_cut, prefix="KA"))
    experiment_df["kurtosis_geomean_box"] = experiment_df[
        "kurtosis_geomean"].apply(
            lambda x: mark_group_name(x, qcut_set=kurtosis_cut, prefix="KG"))
    # kurtosis_ari_freq = experiment_df["kurtosis_arimean_box"].value_counts(
    # ) / experiment_df["kurtosis_arimean_box"].value_counts().sum() * 100
    # kurtosis_ari_freq = sort_Series_by_strkey(input_Series=kurtosis_ari_freq,
    #                                           str_loc=2)
    kurtosis_geo_freq = experiment_df["kurtosis_geomean_box"].value_counts(
    ) / experiment_df["kurtosis_geomean_box"].value_counts().sum() * 100
    kurtosis_geo_freq = sort_Series_by_strkey(input_Series=kurtosis_geo_freq,
                                              str_loc=2)
    plot_frequency_bar(
        freqs={
            # "arithmetic mean": kurtosis_ari_freq,
            "geometric mean": kurtosis_geo_freq
        },
        start_point=0,
        interval=1,
        xticks=np.arange(len(kurtosis_cut) - 1),
        # xticks=np.arange(len(kurtosis_cut) - 1) + 0.2,
        xticklabels=["[0,3]", "(3,10]", "(10,30]", "(30,75]", ">75"],
        xlabel_name="Geometric Averaging Kurtosis",
        output_path=pictures_path,
        picture_name="Fig2B",
        picture_format="tiff",
        annotations={"B": (-0.1, 1.05)},
        # show_label=True,
        bar_width=0.4,
        fig_size=(5,5)
    )


def plot_NIHL_statistic_res(experiment_df, control_df, qcut_set, groupby_key,
                            prefix, start_point, xlabel_name, picture_name,
                            annotations, pictures_path):
    experiment_df[groupby_key + "_box"] = experiment_df[groupby_key].apply(
        lambda x: mark_group_name(x, qcut_set=qcut_set, prefix=prefix))
    experiment_df = experiment_df[experiment_df[
        groupby_key + "_box"].str.startswith(prefix) == True]
    control_df[groupby_key + "_box"] = control_df[groupby_key].apply(
        lambda x: mark_group_name(x, qcut_set=qcut_set, prefix=prefix))
    control_df = control_df[control_df[groupby_key +
                                       "_box"].str.startswith(prefix) == True]
    exper_box_num = experiment_df[groupby_key + "_box"].value_counts()
    exper_NIHL1234_num = experiment_df.groupby(groupby_key +
                                               "_box")["NIHL1234_Y"].sum()
    exper_NIHL346_num = experiment_df.groupby(groupby_key +
                                              "_box")["NIHL346_Y"].sum()
    exper_NIHL1234_freq = exper_NIHL1234_num / exper_box_num * 100
    exper_NIHL1234_freq = sort_Series_by_strkey(
        input_Series=exper_NIHL1234_freq, str_loc=len(prefix))
    exper_NIHL346_freq = exper_NIHL346_num / exper_box_num * 100
    exper_NIHL346_freq = sort_Series_by_strkey(input_Series=exper_NIHL346_freq,
                                               str_loc=len(prefix))

    contr_box_num = control_df[groupby_key + "_box"].value_counts()
    contr_NIHL1234_num = control_df.groupby(groupby_key +
                                            "_box")["NIHL1234_Y"].sum()
    contr_NIHL346_num = control_df.groupby(groupby_key +
                                           "_box")["NIHL346_Y"].sum()
    contr_NIHL1234_freq = contr_NIHL1234_num / contr_box_num * 100
    contr_NIHL1234_freq = sort_Series_by_strkey(
        input_Series=contr_NIHL1234_freq, str_loc=len(prefix))
    contr_NIHL346_freq = contr_NIHL346_num / contr_box_num * 100
    contr_NIHL346_freq = sort_Series_by_strkey(input_Series=contr_NIHL346_freq,
                                               str_loc=len(prefix))

    plot_frequency_bar(freqs={
        "experiment group: $\\text{NIHL}_{1234}$":
        exper_NIHL1234_freq,
        "control group: $\\text{NIHL}_{1234}$":
        contr_NIHL1234_freq,
        "experiment group: $\\text{NIHL}_{346}$":
        exper_NIHL346_freq,
        "control group: $\\text{NIHL}_{346}$":
        contr_NIHL346_freq,
    },
                       start_point=start_point,
                       interval=5,
                       xticks=qcut_set,
                       xticklabels=qcut_set,
                       label_type=False,
                       xlabel_name=xlabel_name,
                       ylabel_name="NIHL proportion (%)",
                       output_path=pictures_path,
                       picture_name=picture_name,
                       picture_format="tiff",
                       annotations=annotations,
                       bar_width=1.0,
                       color_type={
                           "experiment group: $\\text{NIHL}_{1234}$": {
                               "color": "#1f77b4",
                               "hatch": ""
                           },
                           "control group: $\\text{NIHL}_{1234}$": {
                               "color": "#1f77b4",
                               "hatch": "///"
                           },
                           "experiment group: $\\text{NIHL}_{346}$": {
                               "color": "#ff7f0e",
                               "hatch": ""
                           },
                           "control group: $\\text{NIHL}_{346}$": {
                               "color": "#ff7f0e",
                               "hatch": "///"
                           },
                       },
                       userdefine_label={
                           "$\\text{NIHL}_{1234}$":
                           plt.Rectangle((0, 0),
                                         1,
                                         1,
                                         fc="#1f77b4",
                                         ec="black",
                                         alpha=0.4),
                           "$\\text{NIHL}_{346}$":
                           plt.Rectangle((0, 0),
                                         1,
                                         1,
                                         fc="#ff7f0e",
                                         ec="black",
                                         alpha=0.4),
                           "experiment group":
                           plt.Rectangle((0, 0), 1, 1, fc="white", ec="black"),
                           "control group":
                           plt.Rectangle((0, 0),
                                         1,
                                         1,
                                         hatch="///",
                                         fc="white",
                                         ec="black")
                       })


def Mantel_Haenszel_test(experiment_df, control_df, qcut_set, groupby_key, prefix):
    experiment_df[groupby_key + "_box"] = experiment_df[groupby_key].apply(
        lambda x: mark_group_name(x, qcut_set=qcut_set, prefix=prefix))
    experiment_df = experiment_df[experiment_df[
        groupby_key + "_box"].str.startswith(prefix) == True]
    control_df[groupby_key + "_box"] = control_df[groupby_key].apply(
        lambda x: mark_group_name(x, qcut_set=qcut_set, prefix=prefix))
    control_df = control_df[control_df[groupby_key +
                                       "_box"].str.startswith(prefix) == True]
    exper_box_num = experiment_df[groupby_key + "_box"].value_counts()
    exper_NIHL1234_Y_num = experiment_df.groupby(groupby_key +
                                               "_box")["NIHL1234_Y"].sum()
    exper_NIHL1234_N_num = exper_box_num - exper_NIHL1234_Y_num
    exper_NIHL346_Y_num = experiment_df.groupby(groupby_key +
                                              "_box")["NIHL346_Y"].sum()
    exper_NIHL346_N_num = exper_box_num - exper_NIHL346_Y_num

    contr_box_num = control_df[groupby_key + "_box"].value_counts()
    contr_NIHL1234_Y_num = control_df.groupby(groupby_key +
                                            "_box")["NIHL1234_Y"].sum()
    contr_NIHL1234_N_num = contr_box_num - contr_NIHL1234_Y_num
    contr_NIHL346_Y_num = control_df.groupby(groupby_key +
                                           "_box")["NIHL346_Y"].sum()
    contr_NIHL346_N_num = contr_box_num - contr_NIHL346_Y_num

    for group in exper_box_num.index.sort_values():
        #### 构建二元列表
        exposed_1234 = [exper_NIHL1234_Y_num[group], exper_NIHL1234_N_num[group]]
        unexposed_1234 = [contr_NIHL1234_Y_num[group], contr_NIHL1234_N_num[group]]
        exposed_346 = [exper_NIHL346_Y_num[group], exper_NIHL346_N_num[group]]
        unexposed_346 = [contr_NIHL346_Y_num[group], contr_NIHL346_N_num[group]]
        for index, contingency_table in enumerate(([[exposed_1234, unexposed_1234]], [[exposed_346, unexposed_346]])):
            logger.info(f"Mantel-Haenszel test for {group}")
            logger.info(f"Frequency range: {'1234' if index == 0 else '346'}")
            logger.info(f"Contingency table: {contingency_table}")
            #### 执行Mantel-Haenszel校验
            result = sm.stats.StratifiedTable(contingency_table).summary(alpha=0.05, float_format='%.5f', method='normal')
            #### 提取相关结果
            logger.info(f"\n {result}")


if __name__ == "__main__":
    from datetime import datetime
    logger.add(f"./log/Chinese_all_data_statistic_plot-{datetime.now().strftime('%Y-%m-%d')}.log",level="INFO")
    
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_paths",
        type=list,
        default=[
            "./cache/Chinese_extract_experiment_classifier_df.csv",
            "./cache/Chinese_extract_control_classifier_df.csv"
        ])
    parser.add_argument("--output_path", type=str, default="./cache")
    parser.add_argument("--pictures_path", type=str, default="./pictures")
    args = parser.parse_args()

    logger.info("Input Parameters informations:")
    message = "\n".join([f"{k:<20}: {v}" for k, v in vars(args).items()])
    logger.info(message)

    if isinstance(args.input_paths, list):
        input_paths = seq(args.input_paths).map(lambda x: Path(x)).list()
    else:
        input_paths = Path(args.input_paths)
    output_path = Path(args.output_path)
    pictures_path = Path(args.pictures_path)

    for out_path in (output_path, pictures_path):
        if not out_path.exists():
            out_path.mkdir(parents=True)

    for input_path in input_paths:
        if "experiment" in input_path.stem:
            experiment_df = pd.read_csv(input_path,
                                        header=0,
                                        index_col="staff_id")
            experiment_df = experiment_df[experiment_df["LAeq"] >= 70]
        else:
            control_df = pd.read_csv(input_path,
                                     header=0,
                                     index_col="staff_id")

    # experiment group statistic and plot
    ## LAeq frequency
    plot_LAeq_statistic_res(experiment_df=experiment_df,
                            pictures_path=pictures_path)

    ## kurtosis frequency
    plot_kurtosis_statistic_res(experiment_df=experiment_df,
                                pictures_path=pictures_path)

    # experiment group and control group statistic plot
    ## NIHL ratio under age
    plot_NIHL_statistic_res(experiment_df=experiment_df,
                            control_df=control_df,
                            qcut_set=np.arange(15, 65, 5),
                            groupby_key="age",
                            prefix="A",
                            start_point=16,
                            xlabel_name="Age (year)",
                            picture_name="Fig2C",
                            annotations={"C": (-0.1, 1.05)},
                            pictures_path=pictures_path)
    ### Mantel-Haensel Test under age
    Mantel_Haenszel_test(experiment_df=experiment_df, 
                         control_df=control_df,
                         qcut_set=np.arange(15, 65, 5),
                         groupby_key="age",
                         prefix="A")

    ## NIHL ratio under duration
    plot_NIHL_statistic_res(experiment_df=experiment_df,
                            control_df=control_df,
                            qcut_set=np.arange(0, 45, 5),
                            groupby_key="duration",
                            prefix="D",
                            start_point=1,
                            xlabel_name="Duration (year)",
                            picture_name="Fig2D",
                            annotations={"D": (-0.1, 1.05)},
                            pictures_path=pictures_path)
    ### Mantel-Haensel Test under duration
    Mantel_Haenszel_test(experiment_df=experiment_df, 
                         control_df=control_df,
                         qcut_set=np.arange(0, 45, 5),
                         groupby_key="duration",
                         prefix="D")

    print(1)
