import pandas as pd
import sqlite3 as sl

from .decorators import *


@timer
def generate_create_table_sql(dataframe: pd.DataFrame,
                              table_name: str,
                              primary_keys: list = []) -> str:
    """ 生成创建表的 SQL 语句 
    
    Parameters: 
    dataframe (pd.DataFrame): 包含数据的 DataFrame 
    table_name (str): 表名 
    
    Returns: 
    str: 创建表的 SQL 语句 
    """
    # 获取每列的名称和数据类型
    columns_info = []
    for column_name, dtype in dataframe.dtypes.items():
        #将 Pandas 数据类型映射到 SQLite 数据类型
        sqlite_data_type = {
            "int64": "INTEGER",
            "float64": "REAL",
            "object": "TEXT",
            "datetime64[ns]": "TEXT",  # 日期时间类型在 SQLite 中使用 TEXT
        }.get(str(dtype), "TEXT")
        columns_info.append(
            f"{column_name} {sqlite_data_type} PRIMARY KEY"
            if column_name in primary_keys else
            f"{column_name} {sqlite_data_type}")  # 将列信息组合成创建表的 SQL 语句
    create_table_sql = f"CREATE TABLE IF NOT EXISTS {table_name} (\n"
    create_table_sql += ",\n".join(columns_info)
    create_table_sql += "\n);"
    return create_table_sql


@timer
def load_data_from_table(database_path: str,
                         table_name: str,
                         column_names: list = [],
                         filter_condition: dict = {},
                         limit_size: int = 100):
    column_text = "*" if column_names == [] else ",\n\t".join(column_names)
    if filter_condition == {}:
        condition_text = ""
    else:
        condition_text = "WHERE "
        for key, value in filter_condition.items():
            condition_text += f"{key} = '{value}'"
            condition_text += " AND "
        condition_text = condition_text[:-5]
    limit_text = "" if limit_size == 0 else f"LIMIT {limit_size}"

    conn = sl.connect(database=database_path)
    cur = conn.cursor()
    sql_text = f"""SELECT {column_text}
FROM {table_name}
{condition_text}
{limit_text}
    """
    logger.info(f"SQL command:\n{sql_text}")
    
    res = pd.read_sql_query(sql_text, conn)
    cur.close()
    conn.close()
    return res


@timer
def create_table_from_df(database_path: str,
                         dataframe: pd.DataFrame,
                         table_name: str,
                         primary_keys: list = []) -> None:
    create_sql_text = generate_create_table_sql(dataframe=dataframe,
                                                table_name=table_name,
                                                primary_keys=primary_keys)
    conn = sl.connect(database=database_path)
    cur = conn.cursor()
    cur.execute(f"DROP TABLE IF EXISTS {table_name}")
    conn.commit()
    cur.execute(create_sql_text)
    conn.commit()
    dataframe.to_sql(name=table_name,
                     con=conn,
                     if_exists="append",
                     index=False)
    conn.commit()
    cur.close()
    conn.close()
