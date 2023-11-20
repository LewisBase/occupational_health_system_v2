import numpy as np
from loguru import logger
from time import sleep
from functional import seq

from ..point_result import PointResult
# from constants import AuditoryConstants


class PTAResult(PointResult):
    left_ear_data: dict = None
    right_ear_data: dict = None
    better_ear: str = "Right"
    better_ear_data: dict = None
    # better_ear_x: list = None
    # better_ear_y: list = None

    def __init__(self, **data):
        super().__init__(**data)
        # self._build(**data)
        if self.better_ear_data is None:
            self._better_filter(**data)

    def _build(self, **kwargs):
        if self.data is None:
            if self.file_path is not None:
                self._load_from_file()
            else:
                raise ValueError(
                    "parament file_path or data must have one at least")
        else:
            self.x = list(self.data.keys())
            original_y = seq(self.data.values()).map(lambda x: float(
                x) if isinstance(x, (float, int)) else np.nan).list()
            # 不进行修改
            # self.y = original_y
            # PTA的数值应当为5的倍数
            self.y = seq(original_y).map(lambda x: (
                x//5 + 1 if x % 5 >= 3 else x//5) * 5).list()
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

    def _better_filter(self, **data):
        better_ear_strategy = data.get("better_ear_strategy", "mean")
        mean_key = data.get("mean_key", None)

        self.left_ear_data = seq(self.data.items()).filter(lambda x: x[0].startswith(
            "L")).map(lambda x: (int(x[0].split("-")[1]), float(x[1]))).dict()
        self.right_ear_data = seq(self.data.items()).filter(lambda x: x[0].startswith(
            "R")).map(lambda x: (int(x[0].split("-")[1]), float(x[1]))).dict()

        if better_ear_strategy == "mean":
            # 进行更好耳判断时，空值不计入平均值计算中
            left_mean = np.mean(seq(self.left_ear_data.items()).filter(
                lambda x: x[0] in mean_key if mean_key else x[0]).filter(
                    lambda x: not np.isnan(x[1])).map(lambda x: x[1]).list())
            right_mean = np.mean(seq(self.right_ear_data.items()).filter(
                lambda x: x[0] in mean_key if mean_key else x[0]).filter(
                    lambda x: not np.isnan(x[1])).map(lambda x: x[1]).list())
            if left_mean < right_mean:
                self.better_ear = "Left"
                self.better_ear_data = self.left_ear_data.copy()
            else:
                self.better_ear = "Right"
                self.better_ear_data = self.right_ear_data.copy()
        elif better_ear_strategy == "optimum":
            self.better_ear = "Mix"
            better_ear_x = list(seq(self.data.keys()).map(
                lambda x: int(x.split("-")[1])).set())
            better_ear_x.sort()
            better_ear_y = []
            for freq in better_ear_x:
                better_ear_y.append(np.nanmin((self.left_ear_data.get(
                    freq, np.nan), self.right_ear_data.get(freq, np.nan))))
            self.better_ear_data = dict(zip(better_ear_x, better_ear_y))

    def mean(self, **kwargs):
        mean_key = kwargs.get("mean_key", None)
        if not mean_key:
            return np.mean(self.better_ear_y)
        else:
            return np.mean([self.better_ear_data[key] for key in mean_key])
