import numpy as np
import pandas as pd

from pathlib import Path
# from loguru import logger
from functional import seq
from pydantic import BaseModel

from scipy.signal import savgol_filter


class BaseResult(BaseModel):
    data: dict = None
    x: list = None
    y: list = None
    file_path: Path = None
    output_path: Path = None
    file_name: str = None
    file_sep: str = ","
    do_filter: bool = False

    def __init__(self, **data):
        super().__init__(**data)
        self._build(**data)

    def _build(self, **kwargs):
        if self.data is None:
            if self.file_path is not None:
                self._load_from_file()
            else:
                raise ValueError(
                    "parament file_path or data must have one at least")
        else:
            self.x = list(self.data.keys())
            self.y = seq(self.data.values()).map(lambda x: float(
                x) if isinstance(x, (float,int)) else np.nan).list()
            self.data = dict(zip(self.x, self.y))
        if self.do_filter:
            self._filter_signals(**kwargs)

    def _load_from_file(self):
        if self.file_path.suffix == ".xlsx":
            data = pd.read_excel(self.file_path)
        else:
            data = pd.read_csv(self.file_path, sep=self.file_sep)
        self.file_name = self.file_path.stem
        self.x = data.iloc[:, 0].tolist()
        self.y = data.iloc[:, 1].tolist()

    def _filter_signals(self, **kwargs):
        window_length = kwargs.get('window_length', 11)
        polyorder = kwargs.get('polyorder', 1)

        self.y = savgol_filter(
            self.y, window_length=window_length, polyorder=polyorder)

    def mean(self, **kwargs):
        return np.mean(self.y)
