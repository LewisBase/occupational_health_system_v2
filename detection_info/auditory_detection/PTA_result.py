import numpy as np
from loguru import logger
from time import sleep
from functional import seq
from typing import Union

from ..point_result import PointResult


class PTAResult(PointResult):
    left_ear_data: dict = None
    right_ear_data: dict = None
    better_ear: str = "Right"
    better_ear_data: dict = None
    # mean_key: Union[list, dict] = None

    def __init__(self, **data):
        super().__init__(**data)
        # 此时在super中已经执行了一次重构后的_build()
        if self.better_ear_data is None:
            self._better_filter(**data)

    def _build(self, **kwargs):
        PTA_value_fix = kwargs.get("PTA_value_fix", True)
        
        self.x = list(self.data.keys())
        original_y = seq(self.data.values()).map(lambda x: float(
            x) if isinstance(x, (float, int)) else np.nan).list()
        if PTA_value_fix:
            # PTA的数值应当为5的倍数
            self.y = seq(original_y).map(lambda x: (
                x//5 + 1 if x % 5 >= 3 else x//5) * 5).list()
        else:
            # 不进行修改
            self.y = original_y
        if any(np.nan_to_num(self.y) != np.nan_to_num(original_y)):
            logger.warning(
                "ATTENTION: original PTA are not multiple of 5!")
            logger.warning("The Modification program is triggered!")
            logger.warning(f"original value: {original_y}")
            logger.warning(f"modificated value: {self.y}")
            sleep(1)
        self.data = dict(zip(self.x, self.y))
        if self.do_filter:
            self._filter_signals(**kwargs)
            
        

    def _better_filter(self, **kwargs):
        # 有关better_ear_strategy的说明
        # better_mean: 先分别对两边耳的听阈结果计算平均值，再取听阈结果更好的一边作为选择
        # optimum_freq: 先根据两边耳的听阈结果选出指定频域下更好的混合结果，再做平均
        # average_freq: 不做更好耳的判断，直接计算两边耳听阈结果的平均值
        better_ear_strategy = kwargs.get("better_ear_strategy", "mean")
        mean_key = kwargs.get("mean_key", None)
        # 对mean_key进行转换
        if isinstance(mean_key, list):
            mean_key = dict(zip(mean_key, [1/len(mean_key)]*len(mean_key)))
        else:
            mean_key = mean_key
        ## mean_key权重归一化
        mean_key = seq(mean_key.items()).map(lambda x: (x[0], x[1]/sum(mean_key.values()))).dict()

        self.left_ear_data = seq(self.data.items()).filter(lambda x: x[0].startswith(
            "L")).map(lambda x: (int(x[0].split("-")[1]), float(x[1]))).dict()
        self.right_ear_data = seq(self.data.items()).filter(lambda x: x[0].startswith(
            "R")).map(lambda x: (int(x[0].split("-")[1]), float(x[1]))).dict()

        if better_ear_strategy == "better_mean":
            # 进行更好耳判断时，空值不计入平均值计算中
            left_mean = np.mean(seq(self.left_ear_data.items()).filter(
                lambda x: x[0] in mean_key.keys() if mean_key else x[0]).filter(
                    lambda x: not np.isnan(x[1])).map(lambda x: x[1]).list())
            right_mean = np.mean(seq(self.right_ear_data.items()).filter(
                lambda x: x[0] in mean_key.keys() if mean_key else x[0]).filter(
                    lambda x: not np.isnan(x[1])).map(lambda x: x[1]).list())
            if left_mean < right_mean:
                self.better_ear = "Left"
                self.better_ear_data = self.left_ear_data.copy()
            else:
                self.better_ear = "Right"
                self.better_ear_data = self.right_ear_data.copy()
        elif better_ear_strategy == "optimum_freq":
            self.better_ear = "Mix"
            better_ear_x = list(seq(self.data.keys()).map(
                lambda x: int(x.split("-")[1])).set())
            better_ear_x.sort()
            better_ear_y = []
            for freq in better_ear_x:
                better_ear_y.append(np.nanmin((self.left_ear_data.get(
                    freq, np.nan), self.right_ear_data.get(freq, np.nan))))
            self.better_ear_data = dict(zip(better_ear_x, better_ear_y))
        elif better_ear_strategy == "average_freq":
            self.better_ear = "Average"
            better_ear_x = list(seq(self.data.keys()).map(
                lambda x: int(x.split("-")[1])).set())
            better_ear_x.sort()
            better_ear_y = []
            for freq in better_ear_x:
                better_ear_y.append(np.nanmean((self.left_ear_data.get(
                    freq, np.nan), self.right_ear_data.get(freq, np.nan))))
            self.better_ear_data = dict(zip(better_ear_x, better_ear_y))

    def mean(self, **kwargs):
    # 由于再构建对象的过程中就已经指定了better_ear_strategy，所以在调用mean函数时不会再次判断better_ear
        mean_key = kwargs.get("mean_key", None)
        # 对mean_key进行转换
        if isinstance(mean_key, list):
            mean_key = dict(zip(mean_key, [1/len(mean_key)]*len(mean_key)))
        else:
            mean_key = mean_key
        ## mean_key权重归一化
        mean_key = seq(mean_key.items()).map(lambda x: (x[0], x[1]/sum(mean_key.values()))).dict()
        if mean_key:
            mean_key_values = [self.better_ear_data[key]*value for key,value in mean_key.items()]
            return sum(mean_key_values)
        else:
            return np.nanmean(seq(self.better_ear_data.values()).list())
