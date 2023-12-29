import numpy as np
import pandas as pd

from pathlib import Path
# from loguru import logger
from functional import seq
from pydantic import BaseModel

from scipy.signal import savgol_filter


class BaseResult(BaseModel):
    data: dict
    x: list = None
    y: list = None
    # file_path: Path = None
    # output_path: Path = None
    # file_name: str = None
    # file_sep: str = ","
    do_filter: bool = False

    def __init__(self, **data):
        super().__init__(**data)
        self._build(**data)

    def _build(self, **kwargs):
        self.x = list(self.data.keys())
        self.y = seq(self.data.values()).map(lambda x: float(
            x) if isinstance(x, (float,int)) else np.nan).list()
        self.data = dict(zip(self.x, self.y))
        if self.do_filter:
            self._filter_signals(**kwargs)

    @classmethod
    def load_from_file(cls,
                       file_path: str,
                       file_sep: str = ","):
        file_path = Path(file_path)
        if file_path.suffix == ".xlsx":
            data = pd.read_excel(file_path)
        else:
            data = pd.read_csv(file_path, sep=file_sep)
        x = data.iloc[:, 0].tolist()
        y = data.iloc[:, 1].tolist()
        data = dict(zip(x,y))
        return cls(data=data, x=x, y=y)

    def _filter_signals(self, **kwargs):
        window_length = kwargs.get('window_length', 11)
        polyorder = kwargs.get('polyorder', 1)

        self.y = savgol_filter(
            self.y, window_length=window_length, polyorder=polyorder)

    def mean(self, **kwargs):
        return np.mean(self.y)
