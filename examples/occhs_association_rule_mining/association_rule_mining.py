import pandas as pd
import numpy as np
import pickle
from loguru import logger
from functional import seq
from pathlib import Path

from model.data_mining import Apriori
from examples.occhs_time_series_predict.disease_time_series_predict import OCCUPATIONAL_DISEASE_TYPE_NAME

def step(input_path: Path,
         models_path: Path,
         output_path: Path,
         task: str):
    # 加载数据
    input_df = pd.read_csv(input_path, header=0)
    occupational_disease_list = input_df["disease"].drop_duplicates().tolist()

    if task == "train":
        for occupational_disease in occupational_disease_list:
            logger.info(f"Start mining data of {occupational_disease}")
            mining_df = input_df[input_df["disease"] == occupational_disease]
            mining_data = mining_df["hazard"].str.split(",")
            model = Apriori()
            mining_res = model.find_rule(origin_df=mining_data)
            logger.info(f"Mining results: \n {mining_res}")
            pickle.dump(mining_res, open(models_path / f"{OCCUPATIONAL_DISEASE_TYPE_NAME.get(occupational_disease)}_data_mining_model.pkl", "wb"))
            


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str,
                        default="./cache/disease_hazard_group_data.csv")
    parser.add_argument("--output_path", type=str, default="./results")
    parser.add_argument("--models_path", type=str, default="./models")
    parser.add_argument("--task", type=str, default="train")
    args = parser.parse_args()

    input_path = Path(args.input_path)
    output_path = Path(args.output_path)
    models_path = Path(args.models_path)
    task = args.task
    for path in (output_path, models_path):
        if not path.exists():
            path.mkdir(parents=True)

    step(input_path=input_path,
         models_path=models_path, output_path=output_path, task=task)
    print(1)
