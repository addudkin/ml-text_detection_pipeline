import pandas as pd
import torch.nn as nn

from logger.mlflow_tools import MlFlowLogger
from logger.tensorboard_tools import TensorboardLogger
from omegaconf import DictConfig
from typing import List, Dict
import numpy as np


class Logger:
    """Common Logger class for logging training progress, model artifacts, and experiment results.

    Attributes:
        mlflow_log (MlFlowLogger): Instance of MlFlowLogger for logging experiment results to MLFlow.
        tensor_log (TensorboardLogger): Instance of TensorboardLogger for logging training progress to TensorBoard.

    Args:
        config: Configuration dictionary containing settings for logging to MLFlow and TensorBoard.

    Example:
        logger = Logger(config)
        logger.log_epoch_loss(mean_loss, "train", epoch)
        logger.log_model(model, path2mlflow)
    """
    def __init__(
            self,
            config: DictConfig
    ):
        self.mlflow_log = None
        self.tensor_log = None
        if config['main_process']:
            if config['mlflow']:
                self.mlflow_log = MlFlowLogger(**config['mlflow'])
                self.mlflow_log._init_experiment()
                self.mlflow_log.log_experiment_artifacts(
                    config_path=config["config_path"],
                    annotation_folder=config["data"]["annotation_folder"]
                )
            if config['tensorboard']:
                self.tensor_log = TensorboardLogger(**config['tensorboard'])

    def log_epoch_loss(
            self,
            mean_loss: float,
            split: str,
            epoch: int
    ) -> None:
        """Log mean loss for the epoch to both MLFlow and TensorBoard.

        Args:
            mean_loss: Mean loss for the epoch.
            split: Split (e.g. "train", "val") for which the loss was calculated.
            epoch: Number of the epoch.
        """
        if self.mlflow_log:
            self.mlflow_log.write_loss(
                mean_loss,
                split,
                epoch
            )
        if self.tensor_log:
            self.tensor_log.log_scalar(mean_loss, f'{split}/epoch_loss', epoch)

    def log_batch_loss(
            self,
            loss: float,
            split: str,
            step: int
    ) -> None:
        """Log loss for a single batch to TensorBoard.

        Args:
            loss: Loss for the batch.
            split: Split (e.g. "train", "val") for which the loss was calculated.
            step: Number of the batch.
        """
        if self.tensor_log:
            self.tensor_log.log_scalar(loss, f'{split}/batch_loss', step)

    def log_epoch_metrics(
            self,
            metrics: Dict,
            metric_names: List[str],
            epoch: int
    ) -> None:
        """Log metrics for the epoch to both MLFlow and TensorBoard.

        Args:
            metrics: DataFrame containing the metric values for each sample in the epoch.
            metric_names: List of names for the metrics.
            epoch: Number of the epoch.
        """
        if self.mlflow_log:
            self.mlflow_log.write_metrics(metrics, metric_names, epoch)

        if self.tensor_log:
            self.tensor_log.write_metrics(metrics, metric_names, epoch)

    def log_model(
            self,
            path2mlflow: str,
            model: nn.Module = None,
            local_model_path: str = None,
    ) -> None:
        """Log a model to MLFlow or write a model artifact to MLFlow.

        Args:
            path2mlflow: Path to the model in the MLFlow artifact store.
            model: PyTorch model to be logged to MLFlow. Defaults to None.
            local_model_path: Local file path to a model to be written as an artifact to MLFlow. Defaults to None.
        """
        if self.mlflow_log:
            if local_model_path:
                self.mlflow_log.write_artifact(local_model_path, path2mlflow)
            if model:
                self.mlflow_log.register_model(model, path2mlflow)

    def log_lr(
            self,
            value: float,
            step: int
    ) -> None:
        """Log the learning rate to TensorBoard.

        Args:
            value: Current value of the learning rate.
            step: Current training step.
        """
        if self.tensor_log:
            self.tensor_log.log_scalar(value, 'lr', step)