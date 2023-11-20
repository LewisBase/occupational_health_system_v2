import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# from matplotlib.patches import Circle

from .base_result import BaseResult


class PointResult(BaseResult):
    def mean(self, **kwargs):
        mean_key = kwargs.get("mean_key", None)
        if not mean_key:
            return np.mean(self.y)
        else:
            return np.mean([self.data[key] for key in mean_key])