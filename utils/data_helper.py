import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy.stats import ttest_ind_from_stats
from itertools import combinations
import ast
from pathlib import Path
from loguru import logger
from typing import Union, List
from pandas.api.types import is_numeric_dtype
from functional import seq


def load_data(
    input_paths: list,
    sep: str = "\t",
    drop_col: list = ["LAeq_adjust"],
    dropna_set: list = ["NIPTS", "kurtosis_arimean", "kurtosis_geomean"],
    str_filter_dict: dict = {
        "staff_id": ["Waigaoqian", "Dongfeng Reno", "Chunguang", "NSK"]
    },
    num_filter_dict: dict = {
        "age": {
            "up_limit": 60,
            "down_limit": 15
        },
        "LAeq": {
            "up_limit": 100,
            "down_limit": 70
        }
    },
    #   special_filter_dict: dict = {"kurtosis": np.nan, "SPL_dBA": np.nan},
    eval_set: list = ["kurtosis", "SPL_dBA"]
) -> pd.DataFrame:
    """用于加载已经提取好信息，用于后续分析任务的数据

    Args:
        input_paths (list): 提取好信息数据的存放路径，支持多个数据拼接
        sep (str, optional): 分隔符. Defaults to "\t".
        drop_col (list, optional): 需要丢弃的列. Defaults to ["LAeq_adjust"].
        dropna_set (list, optional): 需要去除nan值的列. Defaults to ["NIPTS", "kurtosis_arimean", "kurtosis_geomean"].
        str_filter_dict (_type_, optional): 需要按照字符串内容进行筛选的列及筛选条件. Defaults to {"staff_id": [ "Waigaoqian", "Dongfeng Reno", "Chunguang", "NSK"]}.
        num_filter_dict (_type_, optional): 需要按照数值大小进行筛选的列及筛选条件. Defaults to {"age": {"up_limit": 60, "down_limit": 15}, "LAeq": {"up_limit": 100, "down_limit": 70}}.
        eval_set (list, optional): 需要对存储为字符串的数组、字典进行解析的列. Defaults to ["kurtosis", "SPL_dBA"].

    Returns:
        pd.DataFrame: _description_
    """

    df_total = pd.DataFrame()
    for input_path in input_paths:
        sub_df = pd.read_csv(Path(input_path), sep=sep, header=0)
        df_total = pd.concat([df_total, sub_df], axis=0)
    # step 0. drop invalid column
    if drop_col:
        df_total.drop(drop_col, axis=1, inplace=True)
    # step 1. dropna
    if dropna_set:
        df_total.dropna(subset=dropna_set, inplace=True)
    # step 2. str filter
    if str_filter_dict:
        for key, value in str_filter_dict.items():
            for prefix in value:
                if re.match(r".*-\d+", prefix):
                    df_total = df_total[df_total[key] != prefix]
                else:
                    df_total = df_total[~df_total[key].str.startswith(prefix)]
    # step 3. number filter
    if num_filter_dict:
        for key, subitem in num_filter_dict.items():
            df_total = df_total[(subitem["down_limit"] <= df_total[key])
                                & (df_total[key] <= subitem["up_limit"])]
    # step 4. convert dtype and dropna
    if eval_set:
        for col in eval_set:
            df_total[col] = df_total[col].apply(lambda x: ast.literal_eval(
                x.replace('nan', 'None')) if isinstance(x, str) else x)
            # 去除展开的数组中带有nan的数据
            df_total = df_total[df_total[col].apply(lambda x: not any(
                pd.isna(x)) if isinstance(x, (list, np.ndarray)) else x)]
    # step 5. flter special object
    # 去除展开的数组中带有nan的数据
    # if special_filter_dict:
    #     for key, value in special_filter_dict.items():
    #         df_total = df_total[df_total[key].apply(
    #             lambda x: not any(pd.isna(x)) if isinstance(x, (list, np.ndarray)) else x)]
    # step 5. reset index
    df_total.reset_index(inplace=True, drop=True)
    logger.info(f"Data Size = {df_total.shape[0]}")
    return df_total


# useful util function for data preprocessing
def box_data_multi(df: pd.DataFrame,
                   col: str = "LAeq",
                   groupby_cols: List[str] = ["kurtosis_arimean", "duration"],
                   qcut_sets: List[list] = [[3, 10, 50, np.inf],
                                            [0, 10, 20, np.inf]],
                   prefixs: List[str] = ["K-", "D-"],
                   groupby_func: str = "mean") -> pd.DataFrame:
    """对数据在指定的多个维度上分组后，再按照某一维度进行数据聚合

    Args:
        df (pd.DataFrame): _description_
        col (str, optional): 需要进行聚合的参照维度. Defaults to "LAeq".
        groupby_cols (List[str], optional): 需要分组的多个维度. Defaults to ["kurtosis_arimean", "duration"].
        qcut_sets (List[list], optional): 需要分组维度的分组边界. Defaults to [ [3, 10, 50, np.inf], [0, 10, 20, np.inf]].
        prefixs (List[str], optional): 分组后的编号前缀. Defaults to ["K-", "D-"].
        groupby_func (str, optional): 聚合时使用的方法. Defaults to "mean".

    Returns:
        pd.DataFrame: _description_
    """

    for groupby_col, qcut_set, prefix in zip(groupby_cols, qcut_sets, prefixs):
        df[groupby_col + "-group"] = df[groupby_col].apply(
            lambda x: mark_group_name(x, qcut_set=qcut_set, prefix=prefix))
    group_cols = seq(df.columns).filter(lambda x: x.startswith(
        tuple(groupby_cols)) and x.endswith("-group")).list()
    groups = df.groupby(group_cols)

    df_res = pd.DataFrame()
    for group_name, group_data in groups:
        if (isinstance(group_name, (tuple, list))
                and all([isinstance(name, str) for name in group_name])):
            group_data[col] = group_data[col].astype(int)
            if groupby_func == "mean":
                group_data = group_data.groupby(col).mean(numeric_only=True)
            elif groupby_func == "median":
                group_data = group_data.groupby(col).median(numeric_only=True)
            group_data[col] = group_data.index
            group_data["group_name"] = "+".join(
                [str(name) for name in group_name])
            group_data.reset_index(inplace=True, drop=True)
            df_res = pd.concat([df_res, group_data], axis=0)
        elif isinstance(group_name, str):
            group_data[col] = group_data[col].astype(int)
            if groupby_func == "mean":
                group_data = group_data.groupby(col).mean(numeric_only=True)
            elif groupby_func == "median":
                group_data = group_data.groupby(col).median(numeric_only=True)
            group_data[col] = group_data.index
            group_data["group_name"] = group_name
            group_data.reset_index(inplace=True, drop=True)
            df_res = pd.concat([df_res, group_data], axis=0)

    df_res.reset_index(inplace=True, drop=True)
    logger.info(f"Data Size = {df_res.shape[0]}")
    return df_res


def array_padding(origin_data: Union[list, pd.Series, np.ndarray],
                  constant_values=0) -> np.ndarray:
    """对不等长的数组进行填充

    Args:
        origin_data (Union[list, pd.Series, np.ndarray]): _description_
        constant_values (int, optional): _description_. Defaults to 0.

    Returns:
        np.ndarray: _description_
    """
    max_length = max(len(sublist) for sublist in origin_data)
    padded_array = np.array([
        np.pad(array=sublist,
               pad_width=(0, max_length - len(sublist)),
               constant_values=constant_values) for sublist in origin_data
    ])
    return padded_array


def mark_group_name(x,
                    qcut_set: list = [3, 10, 25, np.inf],
                    prefix: str = "K-") -> str:
    """对数据按照规定的边界条件进行分组

    Args:
        x (_type_): _description_
        qcut_set (list, optional): _description_. Defaults to [3, 10, 25, np.inf].
        prefix (str, optional): _description_. Defaults to "K-".

    Returns:
        str: _description_
    """
    for i in range(len(qcut_set) - 1):
        if qcut_set[i] < x <= qcut_set[i + 1]:
            x = prefix + str(i + 1)
            break
    return x


def _single_group_emm_estimate(df: pd.DataFrame, y_col: str, group_col: str,
                               group_names: list) -> pd.DataFrame:
    """计算组别之间的EMM矩阵

    Args:
        df (pd.DataFrame): _description_
        y_col (str): _description_
        group_col (str): _description_
        group_names (list): _description_

    Returns:
        pd.DataFrame: _description_
    """
    model = sm.OLS.from_formula(f"{y_col}~C({group_col})", data=df)
    result = model.fit()
    emm = result.get_prediction(
        exog=pd.DataFrame({f"{group_col}": group_names})).summary_frame()
    emm.index = group_names
    emm["size"] = df[group_col].value_counts().loc[emm.index]
    return emm


def single_box_boundary_ttest(df: pd.DataFrame, groupby_col: str, y_col: str,
                              qcut_set: list, **kwargs) -> tuple:
    """对分组后的各组EMM进行t检验

    Args:
        df (pd.DataFrame): _description_
        groupby_col (str): _description_
        y_col (str): _description_
        qcut_set (list): _description_

    Returns:
        tuple: _description_
    """
    prefix = kwargs.get("prefix", "K-")

    df = df.copy()

    logger.info(f"boundary type: {qcut_set}")
    # calculate group emm
    df["groupby_" + groupby_col] = df[groupby_col].apply(
        lambda x: mark_group_name(x, qcut_set=qcut_set, prefix=prefix))
    # 获取字符串的分组名称并重新排序
    group_names = seq(df["groupby_" +
                         groupby_col].value_counts().index).filter(
                             lambda x: isinstance(x, str)).list()
    group_names.sort()
    emm_res = _single_group_emm_estimate(df=df,
                                         y_col=y_col,
                                         group_col="groupby_" + groupby_col,
                                         group_names=group_names)
    emm_res["size"] = [
        df[df["groupby_" + groupby_col] == col].shape[0] for col in group_names
    ]
    # 查询分组的边界值
    emm_res["up_boundary"] = seq(emm_res.index).map(
        lambda x: qcut_set[group_names.index(x) + 1]).list()

    # ttest on group
    stats_results = []
    for sample_pair_names in combinations(emm_res.index, 2):
        sample_1 = emm_res.loc[sample_pair_names[0]]
        sample_2 = emm_res.loc[sample_pair_names[1]]
        t_statistic, p_value = ttest_ind_from_stats(mean1=sample_1["mean"],
                                                    std1=sample_1["mean_se"],
                                                    nobs1=sample_1["size"],
                                                    mean2=sample_2["mean"],
                                                    std2=sample_2["mean_se"],
                                                    nobs2=sample_2["size"])
        logger.info(
            f"p = {str(round(p_value,4)) + '*' if p_value<0.05 else str(round(p_value,4))} between {sample_pair_names[0]} and {sample_pair_names[1]}"
        )
        stats_results.append(p_value)

    return emm_res, stats_results


# useful util function for machine learning
def cal_R_square(y: np.ndarray, y_fit: np.ndarray):
    residuals = y - y_fit
    ss_residuals = np.sum(residuals**2)
    ss_total = np.sum((y - np.mean(y))**2)
    R_square = 1 - (ss_residuals / ss_total)
    return R_square


def get_categorical_indicies(X: pd.DataFrame) -> list:
    """获取类别类型的特征列数组

    Args:
        X (pd.DataFrame): _description_

    Returns:
        list: _description_
    """
    cats = []
    for col in X.columns:
        if is_numeric_dtype(X[col]):
            pass
        else:
            cats.append(col)
    cat_indicies = []
    for col in cats:
        cat_indicies.append(X.columns.get_loc(col))
    return cat_indicies


def timeseries_train_test_split(X: pd.DataFrame,
                                y: pd.DataFrame,
                                train_size: float = 0.8):
    train_size = int(train_size * len(X))
    X.sort_index(ascending=True, inplace=True)
    y.sort_index(ascending=True, inplace=True)
    train_X = X.iloc[:train_size]
    test_X = X.iloc[train_size:]
    train_y = y.iloc[:train_size]
    test_y = y.iloc[train_size:]
    return train_X, test_X, train_y, test_y
