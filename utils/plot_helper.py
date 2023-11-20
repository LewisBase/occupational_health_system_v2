import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px

from pathlib import Path
from typing import Union
from functional import seq


# 随机抽取样本并绘制指定列的hist直方分布图
def plot_distribution_hist(df: pd.DataFrame,
                           col: str,
                           subplot_cnt: int = 6,
                           output_path: str = "./pictures/",
                           name: str = "set1",
                           random_seed: int = 42,
                           **kwargs):
    is_show = kwargs.get("is_show", None)

    np.random.seed(random_seed)
    fig, axs = plt.subplots(subplot_cnt // 3, 3, figsize=(12, 6))
    axs = np.ravel(axs)
    indexs = np.random.randint(subplot_cnt) + np.array(
        [df.shape[0] // subplot_cnt * i for i in range(subplot_cnt)])
    for index, ax in zip(indexs, axs):
        ax.hist(df.loc[index][col], bins=50, alpha=0.5, density=True)
        ax.set_xlabel(f"{col}")
        ax.set_title(f"{df.iloc[index]['staff_id']}")
    picture_path = Path(output_path) / f"random-{col}-distribution-{name}.png"
    plt.subplots_adjust(hspace=0.4)
    plt.savefig(picture_path)
    if is_show:
        plt.show()
    # plt.show()
    plt.close(fig=fig)


# 绘制2D/3D散点图
def plot_group_scatter(df: pd.DataFrame,
                       cols: list,
                       groupby_col: str,
                       qcut_set: Union[int, list] = 3,
                       output_path: str = "./pictures/",
                       label_name_dict: dict = {},
                       name: str = "set1",
                       **kwargs):
    is_show = kwargs.get("is_show", None)
    title = kwargs.get("title", None)

    df_sorted = df.sort_values(groupby_col)
    if isinstance(qcut_set, int):
        groups = df_sorted.groupby(
            pd.qcut(df_sorted[groupby_col],
                    q=qcut_set,
                    labels=[groupby_col + str(i) for i in range(qcut_set)]))
    else:
        groups = df_sorted.groupby(
            pd.cut(df_sorted[groupby_col],
                   bins=qcut_set,
                   labels=[
                       groupby_col + str(i) for i in range(len(qcut_set) - 1)
                   ]))

    picture_path = Path(
        output_path
    ) / f"{'_'.join(cols)}-group_{groupby_col}-scatter-{name}.{'png' if len(cols)==2 else 'html'}"

    # colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    if len(cols) == 3:
        fig = go.Figure()
        # for group, color in zip(groups, ):
        for group in groups:
            group_name, group_data = group
            fig.add_trace(
                go.Scatter3d(
                    x=group_data[cols[0]],
                    y=group_data[cols[1]],
                    z=group_data[cols[2]],
                    mode="markers",
                    marker=dict(
                        size=4,
                        # color=color,
                        colorscale='Viridis',
                        opacity=0.8),
                    text=label_name_dict.get(group_name, group_name),
                    hoverinfo='text'  # 鼠标悬停时显示文本
                ))
        fig.update_layout(scene=dict(xaxis=dict(title=cols[0]),
                                     yaxis=dict(title=cols[1]),
                                     zaxis=dict(title=cols[2])))
        fig.write_html(picture_path)
        # fig.show()
        # plot by matplotlib
        # fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        # for group_name, group_data in groups:
        #     ax.scatter(xs=group_data[cols[0]], ys=group_data[cols[1]],
        #                zs=group_data[cols[2]], label=group_name, alpha=0.4)
        # ax.set_xlabel(cols[0])
        # ax.set_ylabel(cols[1])
        # ax.set_zlabel(cols[2])
        # plt.legend(loc="best")
        # plt.savefig(picture_path)
        # plt.show()
        # plt.close(fig=fig)
    else:
        fig, ax = plt.subplots(1, figsize=(6.5, 5))
        for group_name, group_data in groups:
            ax.scatter(x=group_data[cols[0]],
                       y=group_data[cols[1]],
                       label=label_name_dict.get(group_name, group_name),
                       alpha=0.4)
        ax.set_xlabel(label_name_dict.get(cols[0], cols[0]))
        ax.set_ylabel(label_name_dict.get(cols[1], cols[1]))
        plt.legend(loc="best")
        if title:
            plt.title(title)
        plt.savefig(picture_path)
        if is_show:
            plt.show()
        plt.close(fig=fig)


# 绘制逻辑回归的散点与曲线
# ! 目前仅支持一种逻辑回归结果
def plot_logistic_scatter_line(regression_res: dict,
                               output_path: str = "./pictures/",
                               name: str = "set",
                               **kwargs):
    is_show = kwargs.get("is_show", False)
    title = kwargs.get("title")
    ylim = kwargs.get("ylim")

    picture_path = Path(output_path) / f"LAeq_NIPTS-logistic-{name}.png"
    fig, ax = plt.subplots(1, figsize=(6.5, 5))
    for group_name, value in regression_res.items():
        x = value["x"]
        y = value["y"]
        y_fit = value["y_fit"]
        a, b, c = value["params"]
        R_2 = value["R_2"]
        ax.scatter(
            x,
            y,
            label=
            f"{group_name} NIPTS={round(a,1)}/[1+exp({round(b,1)}-LAeq)/{round(c,1)}]: $R^2$={round(R_2,2)}",
            alpha=0.4)
        ax.plot(x, y_fit)
    ax.set_xlabel("$L_{Aeq}$ (dBA)")
    ax.set_ylabel("$NIPTS_{346}$ (dB)")
    if ylim:
        ax.set_ylim(ylim)
    plt.legend(loc="best", fontsize="xx-small")
    if title:
        plt.title(title)
    plt.savefig(picture_path)
    if is_show:
        plt.show()
    plt.close(fig=fig)


# 绘制分组emm结果的条形图
# ! 目前仅支持statsmodels得到的emm结果DataFrame
def plot_emm_group_bar(df: pd.DataFrame,
                       groupby_col: str,
                       output_path: str = "./pictures/",
                       name: str = "set1",
                       **kwargs):
    is_show = kwargs.get("is_show", None)

    fig, ax = plt.subplots(1, figsize=(6.5, 5))
    x_ticks = df.index.tolist()
    y = df["mean"].values
    y_err = df["mean_se"].values
    labels = df["size"].map(lambda x: "n = " + str(x)).tolist()
    up_boundary = "_".join(df["up_boundary"].astype(str).values)
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    bars = ax.bar(x=range(len(x_ticks)),
                  height=y,
                  yerr=y_err,
                  align="center",
                  alpha=0.4,
                  ecolor="black",
                  capsize=5)
    for i in range(len(bars)):
        bars[i].set_color(colors[i])
        ax.text(bars[i].get_x() + bars[i].get_width() / 2,
                bars[i].get_height() / 2,
                labels[i],
                ha='center',
                va='bottom')

    ax.set_xlabel("Kurtosis Category")
    ax.set_xticks(range(len(x_ticks)))
    ax.set_xticklabels(x_ticks)
    ax.set_ylabel("EMM of $NIPTS_{346}$ (dB)")
    picture_path = Path(
        output_path
    ) / f"emm_NIPTS_-group_{groupby_col}-{up_boundary}-bar-{name}.png"
    plt.savefig(picture_path)
    if is_show:
        plt.show()
    plt.close(fig=fig)


# 绘制聚类结果的雷达图
def plot_cluster_radar(model,
                       n_cluster: int,
                       cols: list,
                       name_dict: dict = {},
                       output_path: str = "./pictures/",
                       name: str = "cluster",
                       **kwargs):
    is_show = kwargs.get("is_show", None)

    plot_datas = model.cluster_centers_
    labels = seq(cols).map(lambda x: name_dict.get(x, x))

    # 设置角度
    angles = np.linspace(0, 2 * np.pi, len(cols), endpoint=False)
    # 闭合
    angles = np.concatenate((angles, [angles[0]]))

    fig, ax = plt.subplots(figsize=(10, 10),
                           subplot_kw={"polar": True})  # polar参数为True即极坐标系
    for i in range(n_cluster):
        plot_data = plot_datas[i]
        plot_data = np.concatenate((plot_data, [plot_data[0]]))
        ax.plot(angles, plot_data, 'o-', label='class-' + str(i), linewidth=2)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    # ax.set_title("Cluster Rader Chart")
    plt.legend(loc="best")  # 设置图例位置
    picture_path = Path(output_path) / f"radar-plot-{name}.png"
    plt.savefig(picture_path)
    if is_show:
        plt.show()
    plt.close(fig=fig)


# 绘制降维后的聚类结果
# ! 目前仅支持prince降维后产生的数据格式
def plot_pca_2d(df: pd.DataFrame,
                name: str = "PCA Space",
                output_path: str = "./pictures",
                **kwargs):
    """
    2个主成分的降维可视化
    """
    is_show = kwargs.get("is_show", None)
    picture_path = Path(output_path) / f"pca-2d-{name}.png"

    df = df.astype({"cluster": "object"})  # 指定字段的数据类型
    df = df.sort_values("cluster")

    # 绘图
    fig, ax = plt.subplots(1, figsize=(6.5, 5))
    for cluster in df["cluster"].value_counts().index:
        df_plot = df[df["cluster"] == cluster]
        ax.scatter(df_plot["comp1"],
                   df_plot["comp2"],
                   alpha=0.4,
                   label=f"class:{cluster}")
    plt.legend(loc="best", fontsize="small")
    plt.title(name)
    plt.savefig(picture_path)
    if is_show:
        plt.show()
    plt.close(fig=fig)
