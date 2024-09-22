# -*- coding: utf-8 -*-
"""
@DATE: 2024-09-12 10:09:24
@Author: Liu Hengjiang
@File: examples\\NOISH_1998_studying-01_26\\plot_kurtosis_cluster_boundary.py
@Software: vscode
@Description:
        在噪声峰度的维度上进行聚类并绘制结果
"""

import re
import ast
import pickle
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
from pathlib import Path
from functional import seq
from sklearn.cluster import KMeans 
from sklearn.preprocessing import StandardScaler 
from pyod.models.ecod import ECOD
import prince
from yellowbrick.cluster import KElbowVisualizer
from loguru import logger

from matplotlib.font_manager import FontProperties
from matplotlib import rcParams

config = {
            "font.family": "serif",
            "font.size": 12,
            "mathtext.fontset": "stix",# matplotlib渲染数学字体时使用的字体，和Times New Roman差别不大
            "font.serif": ["STZhongsong"],# 华文中宋
            "axes.unicode_minus": False # 处理负号，即-号
         }
rcParams.update(config)

 
def plot_elbow_res(df_cluster: pd.DataFrame,
                   pictures_path: Path,
                   picture_name: str, 
                   picture_format: str,
                   dpi: int,
                   annotations: dict):
    km = KMeans(init="k-means++", random_state=42, n_init="auto")
    fig, ax = plt.subplots(1, figsize=(6.5, 5))
    visualizer = KElbowVisualizer(km, k=(2, 10))
    visualizer.fit(df_cluster)
    visualizer.finalize()
    ax = plt.gca()
    for label, (x, y) in annotations.items():
        ax.annotate(label,
                        xy=(1, 0),
                        xycoords='axes fraction',
                        xytext=(x, y),
                        textcoords='axes fraction',
                        fontproperties=FontProperties(size=20, weight='bold'))
    picture_path = Path(pictures_path) / f"{picture_name}.{picture_format}"
    plt.savefig(picture_path, format=picture_format, dpi=dpi)

def plot_boundary_res(df_set: pd.DataFrame,
                      labels,
                      boundary: list,
                      pictures_path: Path, 
                      picture_name: str, 
                      picture_format: str, 
                      dpi: int,
                      annotations: dict):
    fig, ax = plt.subplots(1, figsize=(6.5, 5), dpi=dpi)
    for label in set(labels):
        df_plot = df_set.query("labels==@label")
        ax.scatter(df_plot.kurtosis_geomean, df_plot.LAeq, alpha=.3, label=f"class: {label}")
    ymin, ymax = ax.get_ylim()
    ax.vlines(x=boundary[0], ymin=ymin, ymax=ymax, colors="black", linestyles="--")
    ax.vlines(x=boundary[1], ymin=ymin, ymax=ymax, colors="black", linestyles="--")
    ax.vlines(x=boundary[2], ymin=ymin, ymax=ymax, colors="black", linestyles="--")
    ax.vlines(x=boundary[3], ymin=ymin, ymax=ymax, colors="black", linestyles="--")
    ax.annotate(text="KG-1", xy=(-3, 123), color="red")
    ax.annotate(text="KG-2", xy=(24, 120), color="red")
    ax.annotate(text="KG-3", xy=(99, 117), color="red")
    ax.set_ylim(ymin, ymax)
    ax.set_xlabel("kurtosis geometric mean")
    ax.set_ylabel("$L_{Aeq,8h}$ (dBA)")
    ax.set_xticks(boundary + [100, 200, 250, 300, 350])
    for label, (x, y) in annotations.items():
        ax.annotate(label,
                        xy=(1, 0),
                        xycoords='axes fraction',
                        xytext=(x, y),
                        textcoords='axes fraction',
                        fontproperties=FontProperties(size=20, weight='bold'))
    plt.legend(loc="best")
    picture_path = Path(pictures_path) / f"{picture_name}.{picture_format}"
    plt.savefig(picture_path, format=picture_format, dpi=dpi)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, default="./cache/Chinese_extract_experiment_df_average_freq_1234w.csv")
    parser.add_argument("--models_path", type=str, default="./models")
    parser.add_argument("--pictures_path", type=str, default="./pictures")
    # parser.add_argument("--task_name", type=str, default="cluster")
    parser.add_argument("--task_name", type=str, default="plot")
    args = parser.parse_args()
    
    logger.info("Input Parameters informations:")
    message = "\n".join([f"{k:<20}: {v}" for k, v in vars(args).items()])
    logger.info(message)
    
    input_path = Path(args.input_path)
    models_path = Path(args.models_path)
    pictures_path= Path(args.pictures_path)
    task_name = args.task_name
    for out_path in (models_path, pictures_path):
        if not out_path.exists():
            out_path.mkdir(parents=True)
            
    extract_df = pd.read_csv(input_path, header=0, index_col="staff_id")
    useful_cols = ["kurtosis_geomean"]
    df_set = extract_df.query("LAeq >= 70")
    df_cluster = df_set[useful_cols].copy()
    numerical_cols = ["kurtosis_geomean"]
    scaler = StandardScaler()
    df_cluster[numerical_cols] = scaler.fit_transform( df_cluster[numerical_cols])

    if task_name == "cluster":
        # best Kmean model
        n_cluster = 4
        kmeans = KMeans(n_clusters=n_cluster)
        kmeans.fit(df_cluster)
        clusters_predict = kmeans.fit_predict(df_cluster)
        pickle.dump(kmeans, open(models_path / Path("kurtosis_cluster_model.pkl"), "wb"))
        labels = kmeans.labels_
        cluster_centers = kmeans.cluster_centers_
        # cluster center calculate
        class_center_value = {}
        for col in numerical_cols:
            feature_no = df_cluster.columns.tolist().index(col)
            original_center_value = cluster_centers[:, feature_no] * df_set[
                col].std() + df_set[col].mean()
            class_center_value[col] = dict(
                zip(["class-" + str(i) for i in range(n_cluster)],
                    original_center_value))

        df_center = pd.DataFrame.from_dict(class_center_value)
        logger.info(f"{df_center}")
    
    if task_name == "plot":
        # K value recognition
        plot_elbow_res(df_cluster=df_cluster,
                       pictures_path=pictures_path,
                       picture_name = "Fig6A",
                       picture_format = "tiff",
                       dpi = 330,
                       annotations = {"A": (-0.1, 1.05)})
    

        # best Kmean model
        kmeans = pickle.load(open(models_path / Path("kurtosis_cluster_model.pkl"), "rb"))
        labels = kmeans.labels_
        cluster_centers = kmeans.cluster_centers_
        df_set["labels"] = labels

        # plot cluster boundary
        plot_boundary_res(df_set=df_set,
                          labels=labels,
                          boundary=[3, 25, 65, 165],
                          pictures_path=pictures_path,
                          picture_name="Fig6B",
                          picture_format="tiff",
                          dpi=330,
                          annotations={"B":(-0.1, 1.05)})

    print(1)
