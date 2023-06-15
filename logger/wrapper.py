import torch

import pandas as pd
import torch.nn as nn

from logger.tensorboard_tools import TensorboardLogger
from omegaconf import DictConfig
from typing import List, Union


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
        self.tensor_log = None

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
            metrics: List[float],
            metric_names: List[str],
            epoch: int
    ) -> None:
        """Log metrics for the epoch to both MLFlow and TensorBoard.

        Args:
            metrics: DataFrame containing the metric values for each sample in the epoch.
            metric_names: List of names for the metrics.
            epoch: Number of the epoch.
        """
        if self.tensor_log:
            self.tensor_log.write_metrics(metrics, metric_names, epoch)

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
