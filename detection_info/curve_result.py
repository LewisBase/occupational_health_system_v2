import numpy as np
# import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

from pathlib import Path
from loguru import logger
from functional import seq

from scipy.interpolate import CubicSpline, lagrange
from scipy.misc import derivative
from scipy.optimize import curve_fit
from scipy.signal import find_peaks

from .base_result import BaseResult


def curvature(x, func):
    """已知函数形式计算曲率

    Args:
        x (_type_): _description_
        func (_type_): _description_

    Returns:
        _type_: _description_
    """
    func_p = derivative(func, x, dx=1e-6, n=1)
    func_p_p = derivative(func, x, dx=1e-6, n=2)
    curvature = np.abs(func_p_p) / ((1 + func_p**2)**1.5)
    return curvature


def curvature_NLLS(point_x: np.array, point_y: np.array):
    if len(point_x) < 3:
        logger.error(f"length of point_x: {len(point_x)} is too short!")
        raise ValueError("point_x is too short!")

    def model_func(x, a, b, c):
        return a * np.exp(-b * x) + c
    popt, pcov = curve_fit(model_func, point_x, point_y, maxfev=100000)

    def res(x):
        return popt[0] * np.exp(-popt[1] * x) + popt[2]

    point_index = [len(point_x)//2-1, len(point_x)//2, len(point_x)//2+1]
    return curvature(point_x[point_index], res)


def curvature_LGP(point_x: np.array, point_y: np.array):
    if len(point_x) < 3:
        logger.error(f"length of point_x: {len(point_x)} is too short!")
        raise ValueError("point_x is too short!")

    res = lagrange(point_x, point_y)

    point_index = [len(point_x)//2-1, len(point_x)//2, len(point_x)//2+1]
    return curvature(point_x[point_index], res)


def curvature_ND(point_x: np.array, point_y: np.array):
    if len(point_x) < 3:
        logger.error(f"length of point_x: {len(point_x)} is too short!")
        raise ValueError("point_x is too short!")

    h = (max(point_x)-min(point_y))/len(point_x)
    dy = np.gradient(point_y, h)
    d2y = np.gradient(dy, h)
    curvature_res = d2y / (1 + dy**2)**(3/2)

    point_index = [len(point_x)//2-1, len(point_x)//2, len(point_x)//2+1]
    return list(curvature_res[point_index])


def curvature_CS(point_x: np.array, point_y: np.array):
    if len(point_x) < 3:
        logger.error(f"length of point_x: {len(point_x)} is too short!")
        raise ValueError("point_x is too short!")

    res = CubicSpline(point_x, point_y)

    point_index = [len(point_x)//2-1, len(point_x)//2, len(point_x)//2+1]
    return curvature(point_x[point_index], res)


class CurveResult(BaseResult):
    peaks: list = []
    peak_curvature: dict = {}
    peak_left_curvature: dict = {}
    peak_right_curvature: dict = {}
    peak_latency: dict = {}
    peak_half_width: dict = {}
    peak_absolute_amplitude: dict = {}
    peak_left_amplitude: dict = {}
    peak_right_amplitude: dict = {}

    def __init__(self, **data):
        super().__init__(**data)
        self._build(**data)
        self._find_peaks(**data)

    def _find_peaks(self, **kwargs):
        prominence = kwargs.get("prominence", 0.05)
        plot = kwargs.get("plot", False)
        plot_show = kwargs.get("plot_show", False)

        peaks, _ = find_peaks(self.y, prominence=prominence)
        self.peaks = peaks

        if plot or plot_show:
            plt.figure()
            plt.plot(self.x, self.y)
            for peak in self.peaks:
                plt.scatter(self.x[peak], self.y[peak], color="red")
            figure_output_path = self.output_path / Path("peaks")
            if not figure_output_path.exists():
                figure_output_path.mkdir(parents=True)
            plt.savefig(figure_output_path /
                        Path(f"{self.file_path.stem}_peaks.png"))
            if plot_show:
                plt.show()
            plt.close()

    def _amend_peaks(self, amend_peaks: list):
        self.peaks = amend_peaks

    def _find_key_points_for_curvature(self, nums: int = 5, peak_No: int = 1, **kwargs):
        fit_x = kwargs.get("fit_x", len(self.x))

        peak_index = self.peaks[peak_No-1]
        results = []
        results.append((self.x[peak_index], self.y[peak_index]))
        for i in range(1, nums//2+1):
            # 取顶点两侧x轴固定间隔的关键点位置
            gap_x = len(self.x) // fit_x * i
            results.append((self.x[peak_index-gap_x],
                           self.y[peak_index-gap_x]))
            results.append((self.x[peak_index+gap_x],
                           self.y[peak_index+gap_x]))
        return seq(set(results)).sorted(lambda x: x[0]).list()

    def cal_peak_curvature_info(self, func_name: str = "CS", peak_No: int = 1, **kwargs):
        nums = kwargs.get("nums", 5)
        plot = kwargs.get("plot", False)
        plot_show = kwargs.get("plot_show", False)

        CUR_FUNC = {
            "NLLS": curvature_NLLS,
            "LGP": curvature_LGP,
            "ND": curvature_ND,
            "CS": curvature_CS
        }
        key_points = np.array(self._find_key_points_for_curvature(
            nums=nums, peak_No=peak_No, **kwargs))
        cal_res = CUR_FUNC[func_name](
            key_points[:, 0], key_points[:, 1])
        self.peak_left_curvature[peak_No] = cal_res[0]
        self.peak_curvature[peak_No] = cal_res[1]
        self.peak_right_curvature[peak_No] = cal_res[2]

        if plot or plot_show:
            fig, ax = plt.subplots()
            ax.plot(self.x, self.y)
            # ax.plot(key_points[:,0], key_points[:,1])
            for key_point in key_points:
                ax.scatter(key_point[0], key_point[1], color="red")
            center = key_points[len(key_points)//2]
            radius = 1 / cal_res[1]
            center[1] -= radius
            circle = Circle(center, radius, edgecolor='black',
                            facecolor='none')
            ax.add_patch(circle)
            figure_output_path = self.output_path / Path("curvature")
            if not figure_output_path.exists():
                figure_output_path.mkdir(parents=True)
            plt.savefig(figure_output_path /
                        Path(f"{self.file_path.stem}_peak_curvature.png"))
            if plot_show:
                plt.show()
            plt.close()

    def cal_peak_coordinate_info(self, func_name: str = "CS", peak_No: int = 1, **kwargs):
        inter_dense = kwargs.get("inter_dense", 10)
        plot = kwargs.get("plot", False)
        plot_show = kwargs.get("plot_show", False)

        CUR_FUNC = {
            "LGP": lagrange,
            "CS": CubicSpline
        }

        peak_x = self.x[0 if peak_No-2 < 0 else self.peaks[peak_No-2]
            : None if peak_No >= len(self.peaks) else self.peaks[peak_No]]
        peak_y = self.y[0 if peak_No-2 < 0 else self.peaks[peak_No-2]
            : None if peak_No >= len(self.peaks) else self.peaks[peak_No]]
        interpolate = CUR_FUNC[func_name](peak_x, peak_y)

        new_x = np.linspace(min(peak_x), max(peak_x),
                            inter_dense * len(peak_x))
        new_y = interpolate(new_x)
        # ! 如果出现第二峰高于第一峰的情况，取最大值的方法会失效
        # max_index = np.argmax(new_y)
        # ! 用与原数据中最接近峰值的点来判断峰的位置
        max_index = np.argmin(np.abs(new_y - self.y[self.peaks[peak_No-1]]))
        half_max = new_y[max_index] / 2

        # 确认计算峰高的基线位置
        left_min_index = np.argmin(new_y[:max_index])
        right_min_index = np.argmin(new_y[max_index:]) + max_index
        # 确认计算半峰宽的位置
        # ! 在多个峰值存在时，此处需要判断半峰宽的交点是否在当前计算的峰上，不能简单使用argmin进行计算
        # ! 此处改用判断曲线上的值与半峰高度差值正负号变化的方法判断半峰宽所对应的位置
        try:
            left_index = np.where(
                np.diff(np.sign(new_y[:max_index] - half_max)) == 2)[0][-1]
            right_index = np.where(
                np.diff(np.sign(new_y[max_index:] - half_max)) == -2)[0][0] + max_index
        except IndexError:
            logger.error("检索半峰宽左右点时超出索引范围！")
            right_index, left_index = 0, 0
        half_width_res = (right_index - left_index) * \
            (max(peak_x)-min(peak_x))/(inter_dense*len(peak_x))

        self.peak_latency[peak_No] = new_x[max_index] - new_x[0]
        self.peak_half_width[peak_No] = half_width_res
        self.peak_absolute_amplitude[peak_No] = new_y[max_index]
        self.peak_right_amplitude[peak_No] = new_y[max_index] - \
            new_y[right_min_index]
        self.peak_left_amplitude[peak_No] = new_y[max_index] - \
            new_y[left_min_index]

        if plot or plot_show:
            plt.figure()
            plt.plot(self.x, self.y, label="original")
            plt.plot(new_x, new_y, label="interpolated",
                     linestyle="--", color="green")

            # plot peak half width
            plt.hlines(y=half_max, xmin=new_x[left_index],
                       xmax=new_x[right_index], colors="red", linestyles="--")
            # plot peak amplitude
            plt.vlines(x=new_x[max_index], ymin=0,
                       ymax=new_y[max_index], colors="red", linestyles="--")
            plt.vlines(x=new_x[left_min_index], ymin=new_y[left_min_index],
                       ymax=new_y[max_index], color="red", linestyles="--")
            plt.vlines(x=new_x[right_min_index], ymin=new_y[right_min_index],
                       ymax=new_y[max_index], color="red", linestyles="--")
            # plot peak min-max range
            plt.hlines(y=new_y[max_index], xmin=new_x[0],
                       xmax=new_x[-1], colors="black", linestyles="--")
            plt.hlines(y=0, xmin=new_x[0], xmax=new_x[-1],
                       colors="black", linestyles="--")

            plt.scatter(new_x[max_index], new_y[max_index], color="red")
            figure_output_path = self.output_path / Path("coordinate")
            if not figure_output_path.exists():
                figure_output_path.mkdir(parents=True)
            plt.savefig(figure_output_path /
                        Path(f"{self.file_path.stem}_peak_coordinate.png"))
            if plot_show:
                plt.show()
            plt.close()
