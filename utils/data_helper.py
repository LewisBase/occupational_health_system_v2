import re
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error
from scipy.stats import ttest_ind_from_stats
from itertools import combinations
import ast
from pathlib import Path
from loguru import logger
from typing import Union, List
from pandas.api.types import is_numeric_dtype
from functional import seq

from .decorators import *


# reduce memory
@timer
def reduce_mem_usage(df,vervose=True):
    start_mem = df.memory_usage().sum()/1024**2
    numerics = ['int16','int32','int64','float16','float32','float64']

    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)

    end_mem = df.memory_usage().sum()/1024**2
    logger.info('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    logger.info('Decreased by {:.1f}%'.format((start_mem-end_mem)/start_mem*100))
    return df


# @timer
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


@timer
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


@timer
def drop_duplicate_columns(df: pd.DataFrame) -> pd.DataFrame:
    """删除具有相同内容的列

    Args:
        df (pd.DataFrame): _description_

    Returns:
        pd.DataFrame: _description_
    """
    cols = df.columns
    drop_df = df.copy()
    skip_cols = []
    for i in range(len(cols)):
        for j in range(i+1, len(cols)):
            if df[cols[i]].equals(df[cols[j]]):
                logger.info(f"The contents in column {cols[i]} and column {cols[j]} are equal!")
                if cols[j] in drop_df.columns:
                    decision = input(f"Drop the columns {cols[j]}: y or n\n")
                    if decision == "y":
                        logger.info(f"Drop the columns {cols[j]}!")
                        drop_df.drop(cols[j], axis=1, inplace=True)
                    else:
                        skip_cols.append(cols[j])
                else:
                    logger.info(f"The column {cols[j]} has already been droped.")
    logger.info(f"total {len(cols)-len(drop_df.columns)} columns have been droped, {len(skip_cols)} columns have been skipped")
    logger.info(f"Droped columns: {set(cols) - set(drop_df.columns)}")
    logger.info(f"Skipped columns: {skip_cols}")
    return drop_df

    
@timer
def drop_unique_value_columns(df: pd.DataFrame) -> pd.DataFrame:
    """删除整列内容完全一致的列

    Args:
        df (pd.DataFrame): _description_

    Returns:
        pd.DataFrame: _description_
    """
    cols = df.columns
    drop_df = df.copy()
    skip_cols = []
    for col in cols:
        value_counts = df[col].value_counts()
        if value_counts.shape[0] == 1 and value_counts[0] == df.shape[0]:
            logger.info(f"The contents in column {col} are all same!")
            decision = input(f"Drop the columns {col}: y or n\n")
            if decision == "y":
                logger.info(f"Drop the columns {col}!")
                drop_df.drop(col, axis=1, inplace=True)
            else:
                skip_cols.append(col)
    logger.info(f"total {len(cols)-len(drop_df.columns)} columns have been droped, {len(skip_cols)} columns have been skipped")
    logger.info(f"Droped columns: {set(cols) - set(drop_df.columns)}")
    logger.info(f"Skipped columns: {skip_cols}")
    return drop_df


@timer
def timeseries_train_test_split(X: pd.DataFrame,
                                y: pd.DataFrame,
                                train_size: float = 0.8):
    train_size = int(train_size * len(X))
    train_X = X.iloc[:train_size]
    test_X = X.iloc[train_size:]
    train_y = y.iloc[:train_size]
    test_y = y.iloc[train_size:]
    return train_X, test_X, train_y, test_y


# 距离相关系数计算
def dist(x, y):
    #1d only
    return np.abs(x[:, None] - y)

def d_n(x):
    d = dist(x, x)
    dn = d - d.mean(0) - d.mean(1)[:,None] + d.mean()
    return dn

def dcov_all(x: np.array, y: np.array):
    dnx = d_n(x)
    dny = d_n(y)
    
    denom = np.product(dnx.shape)
    dc = (dnx * dny).sum() / denom
    dvx = (dnx**2).sum() / denom
    dvy = (dny**2).sum() / denom
    dr = dc / (np.sqrt(dvx) * np.sqrt(dvy))
    return dr

# RMSE    
def root_mean_squared_error(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_pred=y_pred, y_true=y_true))


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

    
@timer
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


@timer
def filter_data(
    df_total: pd.DataFrame,
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
        df_total (pd.DataFrame): 提取好信息的数据，打平为DataFrame
        drop_col (list, optional): 需要丢弃的列.
                                   Defaults to ["LAeq_adjust"].
        dropna_set (list, optional): 需要去除nan值的列. 
                                   Defaults to ["NIPTS", "kurtosis_arimean", "kurtosis_geomean"].
        str_filter_dict (_type_, optional): 需要按照字符串内容进行筛选的列及筛选条件. 
                                   Defaults to {"staff_id": [ "Waigaoqian", "Dongfeng Reno", "Chunguang", "NSK"]}.
        num_filter_dict (_type_, optional): 需要按照数值大小进行筛选的列及筛选条件. 
                                   Defaults to {"age": {"up_limit": 60, "down_limit": 15}, 
                                                "LAeq": {"up_limit": 100, "down_limit": 70}}.
        eval_set (list, optional): 需要对存储为字符串的数组、字典进行解析的列. 
                                   Defaults to ["kurtosis", "SPL_dBA"].

    Returns:
        pd.DataFrame: _description_
    """

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
    # step 5. reset index
    df_total.reset_index(inplace=True, drop=True)
    logger.info(f"Data Size = {df_total.shape[0]}")
    return df_total

    
def single_group_emm_estimate(df: pd.DataFrame, y_col: str, group_col: str,
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