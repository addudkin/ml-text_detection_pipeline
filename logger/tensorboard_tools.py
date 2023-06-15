import pandas as pd
import numpy as np

from tensorboardX import SummaryWriter
from typing import List, Dict


class TensorboardLogger:
    """Utility class for logging data to TensorBoard.

    Args:
        log_dir: The log directory for the TensorBoard SummaryWriter.

    Attributes:
        writer (SummaryWriter): The TensorBoard SummaryWriter instance.

    Example:
        logger = TensorboardLogger("/path/to/logs")
        logger.log_scalar(0.5, "loss", step=0)
        logger.write_metrics(metrics_df, ["accuracy", "loss"], step=1)
    """
    def __init__(
            self,
            log_dir: str
    ):
        self.writer = SummaryWriter(log_dir)

    def log_scalar(
            self,
            value: float,
            name: str,
            step: int
    ) -> None:
        """Log a scalar variable.

        This method logs the given scalar value to TensorBoard under the given name and step.

        Args:
           value: The value to log.
           name: The name of the scalar.
           step: The step at which to log the scalar.
        """
        self.writer.add_scalar(name, value, step)
        self.writer.flush()

    def write_metrics(
            self,
            metrics: List[float],
            metric_names: List[str],
            step: int
    ) -> None:
        """Write metrics from pandas DataFrame.

        This method writes the specified metrics from the given DataFrame to TensorBoard,
        including the mean value for each metric.

        Args:
            metrics: The metrics DataFrame.
            metric_names: The names of the metrics to write.
            step: The step at which to write the metrics.
        """
        for name, metric in zip(metric_names, metrics):
            self.writer.add_scalar(name, metric, step)

        self.writer.flush()


if __name__ == "__main__":
    log_dir = 'logs/exp1'
    tensorbord_logger = TensorboardLogger(log_dir)

    # Если надо залогировать метрику, лосс или любую другую скалярную величину
    tensorbord_logger.log_scalar(
        value=2.28,
        name='F1',
        step=7
    )

    # Если нужно залогировать какие-то метрики представленные в DataFrame
    data = [
        [1.0, 1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0, 0.0]
    ]
    metrics = pd.DataFrame(data=data)

    metrics.columns = ['WAR', 'CAR', 'CER', 'WER']
    metrics.index = ['seria', 'number']  # Сформирует название по каждому филду: метрика/филд

    tensorbord_logger.write_metrics(
        metrics=metrics,
        metric_names=['WAR', 'CAR'],
        step=1
    )


