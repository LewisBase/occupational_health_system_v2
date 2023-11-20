import numpy as np
from ..curve_result import CurveResult
from scipy.signal import find_peaks
from matplotlib import pyplot as plt
from pathlib import Path


class ABRResult(CurveResult):
    def _find_peaks(self, **kwargs):
        prominence = kwargs.get("prominence", 0.05)
        plot = kwargs.get("plot", False)
        plot_show = kwargs.get("plot_show", False)

        # 波I的潜伏期通常在1-2ms的范围
        incubation_index = np.argmin(np.abs(self.x - 1.0))
        peaks, _ = find_peaks(self.y[incubation_index:], prominence=prominence)
        self.peaks = peaks + incubation_index

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
