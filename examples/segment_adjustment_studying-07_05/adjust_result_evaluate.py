# -*- coding: utf-8 -*-
"""
@DATE: 2024-07-11 11:25:27
@Author: Liu Hengjiang
@File: examples\\segment_adjustment_studying-07_05\\adjust_result_evaluate.py
@Software: vscode
@Description:
        对校正后的结果代入ISO1999,Lempert模型进行NIPTS预测结果的EMM测试
"""

import re
import ast
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from functional import seq
from itertools import product
from loguru import logger
from joblib import Parallel, delayed
import torch

from matplotlib.font_manager import FontProperties
from matplotlib import rcParams

config = {
    "font.family": "serif",
    "font.size": 12,
    "mathtext.fontset": "stix",  # matplotlib渲染数学字体时使用的字体，和Times New Roman差别不大
    "font.serif": ["STZhongsong"],  # 华文中宋
    "axes.unicode_minus": False  # 处理负号，即-号
}
rcParams.update(config)

from staff_info import StaffInfo
from model.linear_regression.custom_linear_regression import SegmentAdjustModel
from utils.data_helper import array_padding, mark_group_name, single_group_emm_estimate


def new_segment_adjust(SPL: np.ndarray, kurtosis: np.ndarray, lambdas: float,
                       beta_baseline: int, in_features: int):
    effective_length = min(len(SPL), len(kurtosis), in_features)
    SPL, kurtosis = array_padding(origin_data=[
        SPL[:effective_length],
        kurtosis[:effective_length],
    ],
                                  constant_values=0)
    SPL_p = SPL + lambdas * np.log10(kurtosis / beta_baseline)
    adjust_LAeq = 10 * np.log10(np.mean(10**(SPL_p / 10)))
    return adjust_LAeq


def _extract_data_for_task(data, Lambdas, **additional_set):
    AriLambda, GeoLambda, SegLambda = Lambdas
    mean_key = additional_set.get("mean_key")
    extrapolation = additional_set.get("extrapolation")
    beta_baseline = additional_set.get("beta_baseline")
    in_features = additional_set.get("in_features")

    # kurtosis SPL-dBA 解析
    for key in ["kurtosis", "SPL_dBA"]:
        data[key] = ast.literal_eval(data[key].replace(
            "nan", "None")) if isinstance(data[key], str) else data[key]
        data[key] = seq(data[key]).map(lambda x: x if x else np.nan)
    # PTA 信息重组
    PTA_res_key = [
        "L-500", "L-1000", "L-2000", "L-3000", "L-4000", "L-6000", "L-8000",
        "R-500", "R-1000", "R-2000", "R-3000", "R-4000", "R-6000", "R-8000"
    ]
    PTA_res_dict = seq(data.items()).filter(lambda x: x[0] in PTA_res_key).map(
        lambda x: (x[0], float(x[1]) if re.fullmatch(r"-?\d+(\.\d+)?", str(x[
            1])) else np.nan)).dict()
    data.update({"auditory_detection": {"PTA": PTA_res_dict}})
    # noise 信息重组
    noise_hazard_key = [
        "kurtosis_arimean", "kurtosis_geomean", "LAeq", "kurtosis", "SPL_dBA"
    ]
    noise_hazard_dict = seq(
        data.items()).filter(lambda x: x[0] in noise_hazard_key).dict()
    data.update({"noise_hazard_info": noise_hazard_dict})
    data.update(additional_set)
    # 构建对象
    staff_info = StaffInfo(**data)

    # 重新提取信息
    res = {}
    res["staff_id"] = staff_info.staff_id
    res["sex"] = staff_info.staff_basic_info.sex
    res["age"] = staff_info.staff_basic_info.age
    res["duration"] = staff_info.staff_basic_info.duration
    # label information
    res["NIPTS"] = staff_info.staff_health_info.auditory_diagnose.get("NIPTS")
    res["NIPTS_pred_2013"] = staff_info.NIPTS_predict_iso1999_2013(
        percentrage=50, mean_key=mean_key)
    res["NIPTS_pred_2023"] = staff_info.NIPTS_predict_iso1999_2023(
        percentrage=50, mean_key=mean_key)
    # feature information
    ## L
    res["LAeq"] = staff_info.staff_occupational_hazard_info.noise_hazard_info.LAeq
    ## kurtosis
    res["kurtosis_arimean"] = staff_info.staff_occupational_hazard_info.noise_hazard_info.kurtosis_arimean
    res["kurtosis_geomean"] = staff_info.staff_occupational_hazard_info.noise_hazard_info.kurtosis_geomean
    ## adjust L
    for method, algorithm_code in product([("total_ari", AriLambda),
                                           ("total_geo", GeoLambda)], ["A+n"]):
        staff_info.staff_occupational_hazard_info.noise_hazard_info.cal_adjust_L(
            Lambda=method[1],
            method=method[0],
            algorithm_code=algorithm_code,
            beta_baseline=beta_baseline)
        res[f"L{algorithm_code[0]}eq_adjust_{method[0]}"] = staff_info.staff_occupational_hazard_info.noise_hazard_info.L_adjust[
            method[0]].get(algorithm_code)
    res["LAeq_adjust_segment_new"] = new_segment_adjust(
        SPL=np.array(staff_info.staff_occupational_hazard_info.
                     noise_hazard_info.SPL_dBA),
        kurtosis=np.array(staff_info.staff_occupational_hazard_info.
                          noise_hazard_info.kurtosis),
        lambdas=np.array(SegLambda),
        beta_baseline=beta_baseline,
        in_features=in_features)
    ## NIPTS adjust results
    res["NIPTS_pred_2013_adjust_ari"] = staff_info.NIPTS_predict_iso1999_2013(
        LAeq=res["LAeq_adjust_total_ari"])
    res["NIPTS_pred_2023_adjust_ari"] = staff_info.NIPTS_predict_iso1999_2023(
        LAeq=res["LAeq_adjust_total_ari"], extrapolation=extrapolation)
    res["NIPTS_pred_2013_adjust_geo"] = staff_info.NIPTS_predict_iso1999_2013(
        LAeq=res["LAeq_adjust_total_geo"])
    res["NIPTS_pred_2023_adjust_geo"] = staff_info.NIPTS_predict_iso1999_2023(
        LAeq=res["LAeq_adjust_total_geo"], extrapolation=extrapolation)
    res["NIPTS_pred_2013_adjust_seg"] = staff_info.NIPTS_predict_iso1999_2013(
        LAeq=res["LAeq_adjust_segment_new"])
    res["NIPTS_pred_2023_adjust_seg"] = staff_info.NIPTS_predict_iso1999_2023(
        LAeq=res["LAeq_adjust_segment_new"], extrapolation=extrapolation)
    return res


def extract_data_for_task(df, Lambdas, n_jobs=-1, **additional_set):
    res = Parallel(n_jobs=n_jobs, backend="multiprocessing")(
        delayed(_extract_data_for_task)(
            data=data[1].to_dict(), Lambdas=Lambdas, **additional_set)
        for data in df.iterrows())
    res_df = pd.DataFrame(res)
    return res_df


def emm_res_plot(emm_res_dict,
                 Lambdas,
                 key_group,
                 picture_path,
                 picture_name: str = "emm_res_0"):
    annotations_dict = {
        "kurtosis_arimean": {
            "A": (-0.1, 1.05)
        },
        "kurtosis_geomean": {
            "B": (-0.1, 1.05)
        }
    }
    for version in ("2013", "2023"):
        fig, ax = plt.subplots(1, 1, figsize=(6.5, 5), dpi=330)
        completed_key = []
        name_dict = {
            "NIPTS_pred_2013_diff":
            "Un-adjusted ISO1999:2013 model",
            "NIPTS_pred_2013_adjust_ari_diff":
            "Arimean adjust($\lambda$=%.2f)" % Lambdas[0],
            "NIPTS_pred_2013_adjust_geo_diff":
            "Geomean adjust($\lambda$=%.2f)" % Lambdas[1],
            "NIPTS_pred_2013_adjust_seg_diff":
            "Segment adjust($\lambda$=%.2f)" % Lambdas[2],
            "NIPTS_pred_2023_diff":
            "Un-adjusted Lempert's model",
            "NIPTS_pred_2023_adjust_ari_diff":
            "Arimean adjust($\lambda$=%.2f)" % Lambdas[0],
            "NIPTS_pred_2023_adjust_geo_diff":
            "Geomean adjust($\lambda$=%.2f)" % Lambdas[1],
            "NIPTS_pred_2023_adjust_seg_diff":
            "Segment adjust($\lambda$=%.2f)" % Lambdas[2],
        }
        keys = seq(emm_res_dict.keys()).filter(lambda x: version in x)
        for key in keys:
            if name_dict[key] not in completed_key:
                y = emm_res_dict[key]["mean"].values
                x = range(len(y))
                yerr = emm_res_dict[key]["mean_se"].values
                points = ax.errorbar(x=x,
                                     y=y,
                                     yerr=yerr,
                                     label=name_dict[key],
                                     marker="s",
                                     capsize=4)  # , color=linecolor)
                ax.hlines(y=0,
                          xmin=-1,
                          xmax=3,
                          linestyles="--",
                          color="black",
                          alpha=0.5)
                if name_dict[key].startswith("Un-adjusted"):
                    for i in range(len(emm_res_dict[key]["size"])):
                        size = emm_res_dict[key]["size"][i]
                        ax.text(x=i - 0.2 if i > 0 else i - 0.05,
                                y=y[i] - 1,
                                s=f"n={size}")
            completed_key.append(name_dict[key])
        ax.set_xticks(x)
        ax.set_xticklabels(["KA-1", "KA-2", "KA-3"] if key_group ==
                           "kurtosis_arimean" else ["KG-1", "KG-2", "KG-3"])
        ax.set_xlabel("Kurtosis Category",
                      fontproperties=FontProperties(weight='bold'))
        ax.set_ylabel("EMM of $NIPTS_{346}$ Difference (dB)",
                      fontproperties=FontProperties(weight='bold'))
        ax.set_xlim(-0.1, 2.1)
        ax.set_ylim(-11, 11)
        ax.set_yticks(np.linspace(-10, 10, 9))
        for label, (x, y) in annotations_dict[key_group].items():
            ax.annotate(label,
                        xy=(1, 0),
                        xycoords='axes fraction',
                        xytext=(x, y),
                        textcoords='axes fraction',
                        fontproperties=FontProperties(size=20, weight='bold'))
        ax.set_title(f"Compared with ISO:1999-{version} version")
        plt.legend(loc="best", fontsize="x-small")
        plt.savefig(picture_path / f"{picture_name}-{label}-{version}.png")
        plt.close(fig=fig)


def emm_res_csv(emm_res_dict, Lambdas, key_group, output_path, filename):
    df_save = pd.DataFrame()
    for key, value in emm_res_dict.items():
        value["col_name"] = key
        value["ariLambda"] = Lambdas[0]
        value["geoLambda"] = Lambdas[1]
        value["segLambda"] = Lambdas[2]
        df_save = pd.concat([df_save, value], axis=0)
    df_save.to_csv(output_path / f"{filename}-{key_group}.csv",
                   header=True,
                   index=True)


if __name__ == "__main__":
    from datetime import datetime
    logger.add(
        f"./log/adjust_result_evaluate-{datetime.now().strftime('%Y-%m-%d')}.log",
        level="INFO")

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path",
                        type=str,
                        default="./results/filter_Chinese_extract_df.csv")
    parser.add_argument("--output_path", type=str, default="./results")
    parser.add_argument("--picture_path", type=str, default="./pictures")
    parser.add_argument("--model_path", type=str, default="./models")
    parser.add_argument("--additional_set",
                        type=dict,
                        default={
                            "mean_key": [3000, 4000, 6000],
                            "better_ear_strategy": "optimum_freq",
                            "NIPTS_diagnose_strategy": "better",
                            "extrapolation": "Linear",
                            "beta_baseline": 3,
                            "in_features": 480,
                            "key_group": "kurtosis_arimean",
                            "qcut_set": [3, 10, 50, np.inf],
                            # "key_group": "kurtosis_geomean",
                            # "qcut_set": [3, 10, 25, np.inf],
                            # "Lambdas": [6.5, 6.5, 6.5],
                            # "Lambdas": [5.42, 5.42, 5.42],
                            # "Lambdas": [7.02, 8.30, 5.42],
                            "Lambdas": [6.50, 6.50, 5.42],
                        })
    parser.add_argument("--n_jobs", type=int, default=-1)
    args = parser.parse_args()

    logger.info("Input Parameters informations:")
    message = "\n".join([f"{k:<20}: {v}" for k, v in vars(args).items()])
    logger.info(message)

    input_path = Path(args.input_path)
    output_path = Path(args.output_path)
    picture_path = Path(args.picture_path)
    model_path = Path(args.model_path)
    additional_set = args.additional_set
    n_jobs = args.n_jobs

    # 从additional_set中提取设置
    in_features = additional_set.get("in_features")
    key_group = additional_set.pop("key_group")
    qcut_set = additional_set.pop("qcut_set")
    Lambdas = additional_set.pop("Lambdas")

    # 加载筛选后的样本数据
    filter_df = pd.read_csv(input_path, header=0)

    # # 加载模型结果
    # segment_adjust = SegmentAdjustModel(in_features=in_features,
    #                                     out_features=1)
    # segment_adjust.load_state_dict(
    #     torch.load(model_path / "SegmentAdjust_checkpoint.pt"))
    # hat_lambda = segment_adjust.state_dict()["custom_layer.hat_lambda"].cpu(
    # ).numpy()[0]
    # segLambda = hat_lambda * np.ones(in_features)
    # segLambda = 6.5 * np.ones(in_features)

    # 计算校正后LAeq代入ISO等模型后的结果
    adjust_res_df = extract_data_for_task(df=filter_df,
                                          Lambdas=Lambdas,
                                          n_jobs=n_jobs,
                                          **additional_set)

    # 分组EMM计算
    plot_col = [
        "NIPTS_pred_2013",
        "NIPTS_pred_2013_adjust_ari",
        "NIPTS_pred_2013_adjust_geo",
        "NIPTS_pred_2013_adjust_seg",
        "NIPTS_pred_2023",
        "NIPTS_pred_2023_adjust_ari",
        "NIPTS_pred_2023_adjust_geo",
        "NIPTS_pred_2023_adjust_seg",
    ]
    for col in plot_col:
        adjust_res_df[col +
                      "_diff"] = adjust_res_df["NIPTS"] - adjust_res_df[col]
    adjust_res_df["group_col"] = adjust_res_df[key_group].apply(
        lambda x: mark_group_name(x, qcut_set=qcut_set))
    emm_cols = seq(adjust_res_df.columns).filter(lambda x: x.endswith("diff"))
    df_emm = adjust_res_df[emm_cols + ["group_col"]]

    emm_res_dict = {}
    for col in emm_cols:
        df_emm_cal = df_emm.dropna(subset=[col])
        emm_res = single_group_emm_estimate(df=df_emm_cal,
                                            y_col=col,
                                            group_col="group_col",
                                            group_names=["K-1", "K-2", "K-3"])
        emm_res_dict[col] = emm_res

    ## save results
    emm_res_csv(emm_res_dict=emm_res_dict,
                Lambdas=Lambdas,
                key_group=key_group,
                output_path=output_path,
                filename="emm_res_4")

    ## plot results
    emm_res_plot(emm_res_dict=emm_res_dict,
                 Lambdas=Lambdas,
                 key_group=key_group,
                 picture_path=picture_path,
                 picture_name="emm_res_4")
    print(1)
