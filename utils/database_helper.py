import pandas as pd
import sqlite3


def generate_create_table_sql(dataframe: pd.DataFrame, table_name: str) -> str:
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
            'int64': 'INTEGER',
            'float64': 'REAL',
            'object': 'TEXT',
            'datetime64[ns]': 'TEXT',  # 日期时间类型在 SQLite 中使用 TEXT
        }.get(str(dtype), 'TEXT')
    columns_info.append(
        f'{column_name} {sqlite_data_type}')  # 将列信息组合成创建表的 SQL 语句
    create_table_sql = f'CREATE TABLE IF NOT EXISTS {table_name} (\n'
    create_table_sql += ',\n'.join(columns_info)
    create_table_sql += '\n);'
    return create_table_sql
