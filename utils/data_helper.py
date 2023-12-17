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


@timer
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
