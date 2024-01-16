# -*- coding: utf-8 -*-
"""
@DATE: 2024-01-15 14:38:48
@Author: Liu Hengjiang
@File: examples\MMoE_studying-01_09\SR_regression.py
@Software: vscode
@Description:
        使用PhySo进行符号回归测试
"""

import re
import pickle
import pandas as pd
import numpy as np
import statsmodels.api as sm
import pickle
from joblib import Parallel, delayed
from pathlib import Path
from functional import seq
from itertools import product
from loguru import logger
from tqdm import tqdm

import torch
import torch.nn as nn
import physo
from physo.learn import monitoring
from physo.task import benchmark
from sklearn.model_selection import train_test_split

from staff_info import StaffInfo
from utils.data_helper import box_data_multi

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


def _extract_data_for_task(data: StaffInfo, **additional_set):
    better_ear_strategy = additional_set.pop("better_ear_strategy")
    NIPTS_diagnose_strategy = additional_set.pop("NIPTS_diagnose_strategy")

    res = {}
    res["staff_id"] = data.staff_id
    # label information
    res["NIPTS"] = data.staff_health_info.auditory_diagnose.get("NIPTS")
    # feature information
    ## user features
    res["age"] = data.staff_basic_info.age
    res["sex"] = data.staff_basic_info.sex
    res["duration"] = data.staff_basic_info.duration
    res["work_position"] = data.staff_basic_info.work_position
    ## L
    res["Leq"] = data.staff_occupational_hazard_info.noise_hazard_info.Leq
    res["LAeq"] = data.staff_occupational_hazard_info.noise_hazard_info.LAeq
    ## adjust L
    for method, algorithm_code in product(
        ["total_ari", "total_geo", "segment_ari"], ["A+n"]):
        res[f"L{algorithm_code[0]}eq_adjust_{method}_{algorithm_code}"] = data.staff_occupational_hazard_info.noise_hazard_info.L_adjust[
            method].get(algorithm_code)
    ## kurtosis
    res["kurtosis_geomean"] = data.staff_occupational_hazard_info.noise_hazard_info.kurtosis_geomean
    ## Peak SPL
    res["max_Peak_SPL_dB"] = data.staff_occupational_hazard_info.noise_hazard_info.Max_Peak_SPL_dB
    ## other features in frequency domain

    return res


def extract_data_for_task(df, n_jobs=-1, **additional_set):
    res = Parallel(n_jobs=n_jobs, backend="multiprocessing")(
        delayed(_extract_data_for_task)(data=data, **additional_set)
        for data in df)
    res_df = pd.DataFrame(res)
    return res_df


if __name__ == "__main__":
    from datetime import datetime
    logger.add(
        f"./log/SR_regression-{datetime.now().strftime('%Y-%m-%d')}.log",
        level="INFO")

    import argparse
    parser = argparse.ArgumentParser()
    # parser.add_argument("--task", type=str, default="extract")
    # parser.add_argument("--input_path",
    #                     type=str,
    #                     default="./cache/extract_data.pkl")
    parser.add_argument("--task", type=str, default="analysis")
    parser.add_argument("--input_path",
                        type=str,
                        default="./results/extract_df_SR.csv")
    parser.add_argument("--output_path", type=str, default="./results")
    parser.add_argument("--picture_path", type=str, default="./pictures")
    parser.add_argument("--model_path", type=str, default="./models")
    parser.add_argument("--additional_set",
                        type=dict,
                        default={
                            "mean_key": [3000, 4000, 6000],
                            "better_ear_strategy": "optimum",
                            "NIPTS_diagnose_strategy": "better"
                        })
    parser.add_argument("--n_jobs", type=int, default=-1)
    parser.add_argument("--is_show", type=bool, default=True)
    args = parser.parse_args()

    logger.info("Input Parameters informations:")
    message = "\n".join([f"{k:<20}: {v}" for k, v in vars(args).items()])
    logger.info(message)

    task = args.task
    input_path = Path(args.input_path)
    output_path = Path(args.output_path)
    picture_path = Path(args.picture_path)
    model_path = Path(args.model_path)
    additional_set = args.additional_set
    n_jobs = args.n_jobs
    is_show = args.is_show

    for output in (output_path, picture_path, model_path):
        if not output.exists():
            output.mkdir(parents=True)

    if task == "extract":
        original_data = pickle.load(open(input_path, "rb"))

        extract_df = extract_data_for_task(df=original_data,
                                           n_jobs=n_jobs,
                                           **additional_set)
        extract_df.index = extract_df.staff_id
        extract_df.drop("staff_id", axis=1, inplace=True)
        extract_df.to_csv(output_path / "extract_df_SR.csv",
                          header=True,
                          index=True)

    if task == "analysis":
        extract_df = pd.read_csv(input_path, header=0, index_col="staff_id")
        logger.info(f"Size: {extract_df.shape}")
        extract_df.dropna(how="any", axis=0, inplace=True)
        logger.info(f"Size after dropna: {extract_df.shape}")
        box_df = box_data_multi(df=extract_df,
                                col="LAeq",
                                groupby_cols=["kurtosis_geomean"],
                                qcut_sets=[[3, 10, 25, np.inf]],
                                prefixs=["K-"],
                                groupby_func="mean")

        initial_features_search = [
            "LAeq",
            # "Leq",
            # "max_Peak_SPL_dB",
            "kurtosis_geomean",
        ]
        X = box_df[initial_features_search]
        y = box_df["NIPTS"]

        X_train, X_test, y_train, y_test = train_test_split(X,
                                                            y,
                                                            test_size=0.2,
                                                            random_state=42)

        DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        # Seed
        seed = 42
        np.random.seed(seed)
        torch.manual_seed(seed)
        # ------ Vectors ------
        # Stack of all input variables
        X = torch.tensor(X_train.T.values).to(DEVICE)
        # Output of symbolic function to guess
        y = torch.tensor(y_train.values).to(DEVICE)

        # ------ Constants ------
        const1 = torch.tensor(np.array(1.)).to(DEVICE)
        l = torch.tensor(np.array(6.5)).to(DEVICE)

        args_make_tokens = {
            # operations
            "op_names":
            ["mul", "add", "sub", "div", "inv", "n2", "sqrt", "exp", "log"],
            "use_protected_ops":
            True,
            # input variables
            "input_var_ids": {
                "L": 0,
                "b": 1,
            },
            "input_var_units": {
                "L": [1, 0, 0],
                "b": [0, 0, 0],
            },
            "input_var_complexity": {
                "L": 1.,
                "b": 1.,
            },
            # constants
            "constants": {
                "1": const1,
            },
            "constants_units": {
                "1": [0, 0, 0],
            },
            "constants_complexity": {
                "1": 1.,
            },
            # free constants
            "free_constants": {"l"},
            "free_constants_init_val": {
                "l": 1.
            },
            "free_constants_units": {
                "l": [1, 0, 0]
            },
            "free_constants_complexity": {
                "l": 1.
            },
        }

        library_config = {
            "args_make_tokens": args_make_tokens,
            "superparent_units": [1, 0, 0],
            "superparent_name": "NIPTS",
        }
        reward_config = {
            "reward_function":
            physo.physym.reward.SquashedNRMSE,  # PHYSICALITY
            "zero_out_unphysical": True,
            "zero_out_duplicates": False,
            "keep_lowest_complexity_duplicate": False,
            "parallel_mode": True,
            "n_cpus": None,
        }
        BATCH_SIZE = int(1e3)
        MAX_LENGTH = 7
        GET_OPTIMIZER = lambda model: torch.optim.Adam(
            model.parameters(),
            lr=0.0025,  #0.001, #0.0050, #0.0005, #1,  #lr=0.0025
        )
        learning_config = {
            # Batch related
            'batch_size': BATCH_SIZE,
            'max_time_step': MAX_LENGTH,
            'n_epochs': 100,
            # Loss related
            'gamma_decay': 0.7,
            'entropy_weight': 0.005,
            # Reward related
            'risk_factor': 0.05,
            'rewards_computer': physo.physym.reward.make_RewardsComputer(**reward_config),
            # Optimizer
            'get_optimizer': GET_OPTIMIZER,
            'observe_units': True,
        }
        free_const_opti_args = {
            'loss': "MSE",
            'method': 'LBFGS',
            'method_args': {
                'n_steps': 15,
                'tol': 1e-8,
                'lbfgs_func_args': {
                    'max_iter': 4,
                    'line_search_fn': "strong_wolfe",
                },
            },
        }
        priors_config = [
            #("UniformArityPrior", None),
            # LENGTH RELATED
            ("HardLengthPrior", {
                "min_length": 4,
                "max_length": MAX_LENGTH,
            }),
            ("SoftLengthPrior", {
                "length_loc": 6,
                "scale": 5,
            }),
            # RELATIONSHIPS RELATED
            ("NoUselessInversePrior", None),
            ("PhysicalUnitsPrior", {
                "prob_eps": np.finfo(np.float32).eps
            }),  # PHYSICALITY
            #("NestedFunctions", {"functions":["exp",], "max_nesting" : 1}),
            #("NestedFunctions", {"functions":["log",], "max_nesting" : 1}),
            ("NestedTrigonometryPrior", {
                "max_nesting": 1
            }),
            ("OccurrencesPrior", {
                "targets": [
                    "1",
                ],
                "max": [
                    3,
                ]
            }),
        ]
        cell_config = {
            "hidden_size": 128,
            "n_layers": 1,
            "is_lobotomized": False,
        }
        # 此处按.进行分割
        save_path_training_curves = "pictures/SR_ini_curves.png"
        save_path_log = "log/SR_ini.log"

        run_logger = monitoring.RunLogger(save_path=save_path_log,
                                          do_save=True)

        run_visualiser = monitoring.RunVisualiser(
            epoch_refresh_rate=5,
            save_path=save_path_training_curves,
            do_show=False,
            do_prints=True,
            do_save=True,
        )
        run_config = {
            "learning_config": learning_config,
            "reward_config": reward_config,
            "free_const_opti_args": free_const_opti_args,
            "library_config": library_config,
            "priors_config": priors_config,
            "cell_config": cell_config,
            "run_logger": run_logger,
            "run_visualiser": run_visualiser,
        }

        expression, logs = physo.fit (X, y, run_config,
                                stop_reward = 0.9999, 
                                stop_after_n_epochs = 5)

        pareto_front_complexities, pareto_front_programs, pareto_front_r, pareto_front_rmse = run_logger.get_pareto_front()
        for prog in pareto_front_programs:
            prog.show_infix(do_simplify=True)
            free_consts = prog.free_const_values.detach().cpu().numpy()
            for i in range (len(free_consts)):
                print("%s = %f"%(prog.library.free_const_names[i], free_consts[i]))
    print(1)
