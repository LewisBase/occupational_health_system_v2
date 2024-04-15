import time
import numpy as np
import pandas as pd

from loguru import logger
from typing import Any
from pydantic import BaseModel
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules


class Apriori(BaseModel):
    support: float = 0.3  # 最小支持度
    confidence: float = 0.9  # 最小置信度
    d: Any = None

    def trans_encoder(self, origin_df: pd.DataFrame):
        logger.info("转换原始数据至0-1矩阵...")
        start = time.time()
        te = TransactionEncoder()
        te_ary = te.fit(origin_df.values).transform(origin_df.values)
        self.d = pd.DataFrame(te_ary, columns=te.columns_)
        end = time.time()
        logger.info(f"转换完毕，用时：{round(end-start,2)}秒")
        
    # 寻找关联规则的函数
    def find_rule(self, origin_df: pd.DataFrame):
        self.trans_encoder(origin_df=origin_df)
        logger.info("开始搜索关联规则...")
        start = time.time()
        frq_items = apriori(self.d, min_support=self.support, use_colnames=True)
        rules = association_rules(frq_items, metric="lift", min_threshold=self.confidence)
        frq_items.sort_values(["support"], ascending=False, inplace=True)
        frq_items["itemsets"] = frq_items["itemsets"].apply(lambda x: str(x))
        rules.sort_values(['confidence', 'lift'], ascending=[False, False], inplace=True)
        for col in ["antecedents", "consequents"]:
            rules[col] = rules[col].apply(lambda x: str(x))
        end = time.time()
        logger.info(f"搜索完成，用时：{round(end-start,2)}秒")
        return frq_items, rules
