# -*- coding: utf-8 -*-
"""The module run validation script and save results to mlflow server"""
import os
import json
import logging
from pathlib import Path
import click
import mlflow
import pandas as pd
from dotenv import load_dotenv
from mlflow.tracking.client import MlflowClient
from sklearn.metrics import precision_score, recall_score, roc_auc_score, roc_curve


def find_mlflow_run_id(client, experiment_name: str):
    """
    Find experiment_id and run_id where to save metrics
    Args:
        experiment_name: name of the experiment
    Returns:
        run_id: run to save metrics
    """

    if experiment_name is None:
        experiments = client.list_experiments()
        current_experiment = experiments[-1]
        df_runs = mlflow.search_runs([current_experiment.experiment_id])
        df_runs.sort_values(by="start_time", inplace=True)
        print(df_runs.loc[:, ["run_id", "experiment_id", "start_time"]])
        run_id = df_runs.run_id.values[-2]
        # TODO: need to find run_id in a more precise way
    else:
        current_experiment = mlflow.get_experiment_by_name(experiment_name)
        df_runs = mlflow.search_runs([current_experiment.experiment_id])
        run_id = df_runs.run_id.values[-2]

    return run_id


def save_metrics_for_dvc(
    precision: float,
    recall: float,
    roc_auc: float,
    tpr_fpr: tuple,
    path_to_metrics_storage: str,
) -> None:
    """
    Save dvc metrics
    Args:
        path_to_metrics_storage: path to save metrics

    Returns:
        None
    """

    metrics = {
        "train": {
            "precision": precision,
            "recall": recall,
            "roc_auc": roc_auc,
        }
    }

    plots = {"train": [{"tpr": i, "fpr": j} for i, j in zip(*tpr_fpr)]}

    with open(
        str(path_to_metrics_storage / "metrics.json"),
        "w",
        encoding="UTF-8",
    ) as handler:
        json.dump(metrics, handler)

    with open(
        str(path_to_metrics_storage / "plots.json"),
        "w",
        encoding="UTF-8",
    ) as handler:
        json.dump(plots, handler)


@click.command()
@click.option("--path_to_dataset", default="data/processed/test.csv", type=str)
@click.option("--path_to_metrics_storage", default="reports/metrics", type=str)
@click.option("--registered_model_name", default="default_model", type=str)
@click.option("--experiment_name", default=None)
def main(
    path_to_dataset: str = "data/processed/test.csv",
    path_to_metrics_storage: str = "reports/metrics",
    registered_model_name: str = "default_model",
    experiment_name=None,
) -> None:
    """
    Runs validation method
    Args:
        path_to_dataset: path to test or val dataset
        path_to_metrics_storage: path to save metrics
        registered_model_name: model name registered in mlflow
        experiment_name: experiment name

    Returns:
        None
    """
    client = MlflowClient()

    run_id = find_mlflow_run_id(client, experiment_name)

    with mlflow.start_run(run_id=run_id):
        print(run_id)
        logger = logging.getLogger(__name__)
        logger.info("Start predicting process")

        path_to_dataset = ROOT / Path(path_to_dataset)
        path_to_metrics_storage = ROOT / Path(path_to_metrics_storage)

        # read dataset
        test = pd.read_csv(path_to_dataset).drop(columns=["Unnamed: 0"])
        # TODO: save dataset without that column

        # Now separate the dataset as response variable and feature variables
        x_test = test.drop("target", axis=1)
        y_test = test["target"]

        latest_version = client.get_latest_versions(
            registered_model_name, stages=["None"]
        )[0].version

        clf = mlflow.sklearn.load_model(
            f"models:/{registered_model_name}/{latest_version}"
        )

        predictions = clf.predict(x_test)
        predictions_proba = clf.predict_proba(x_test)

        # Let's see how our model performed
        precision = precision_score(y_test.values, predictions)
        recall = recall_score(y_test.values, predictions)
        roc_auc = roc_auc_score(y_test.values, predictions_proba[:, 1])

        mlflow.log_metrics({"test_precision": precision})
        mlflow.log_metrics({"test_recall": recall})
        mlflow.log_metrics({"test_roc_auc": roc_auc})

        fpr, tpr, _ = roc_curve(y_test.values, predictions_proba[:, 1])

        save_metrics_for_dvc(
            precision, recall, roc_auc, (tpr, fpr), path_to_metrics_storage
        )

        logger.info("done!")


if __name__ == "__main__":
    load_dotenv()
    remote_server_uri = os.getenv("MLFLOW_TRACKING_URI")
    mlflow.set_tracking_uri(remote_server_uri)

    ROOT = Path(__file__).parent.parent.parent

    LOG_FMT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=LOG_FMT)
    main()
