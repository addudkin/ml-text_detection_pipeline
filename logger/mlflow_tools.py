import os
import mlflow
import shutil
import torch.nn as nn
import pandas as pd

from mlflow.tracking.client import MlflowClient
from typing import List, Dict


class MlFlowLogger:
    """A class for logging experiment artifacts to the mlflow server.

    This class allows you to log various artifacts (such as config files, annotations, and model weights) to the
    mlflow server, organized by experiment name and run id. It also provides a method for logging
    various metrics and parameters associated with a run of the experiment.

    MlFlow has the following structure:

    exp_name: <- variable have your own exp_name and experiment_id ∈ [0 - N]
        run_id1 <- unique hash
        run_id2
        .
        .
        .
        run_idN
    exp_name2:
        .
        .

    Args:
        exp_name: The name of the experiment to log to. The desired format is
            "pipeline_name/project_name/your_experiment_name", where:
                - `pipeline_name` is the name of the pipeline (e.g. TR, FieldNet, Clf).
                - `project_name` is the global dbrain project name (e.g. passport, cism)
                - `your_experiment_name` is a short description of the experiment (e.g. exp1, exp2, ... expN).
        username: The username for the mlflow server. Defaults to 'ds'.
        password: The password for the mlflow server. Defaults to 'EitahsaeghoXooz1Hiev'.
        url: The URL for the mlflow server. Defaults to 'https://ml.dbrain.io/'.
        run_id: The run id for the current experiment run. If not provided, a new run
            will be created.

    Attributes:
        client (mlflow.tracking.client.MlflowClient): The mlflow client object for interacting with the
            mlflow server.
        exp_name (str): The name of the experiment to log to.
        run_id (str): The run id for the current experiment run.
    """
    def __init__(
            self,
            exp_name: str,
            username: str = 'ds',
            password: str = 'EitahsaeghoXooz1Hiev',
            url: str = 'https://ml.dbrain.io/',
            run_id: str = None
    ):

        # Init MlFlow client
        self.client = MlflowClient(url)
        self.exp_name = exp_name

        os.environ['MLFLOW_TRACKING_USERNAME'] = username
        os.environ['MLFLOW_TRACKING_PASSWORD'] = password

        self.run_id = run_id

    def _init_experiment(self) -> None:
        """Initialize the current experiment by creating a new run or retrieving the latest run id.

        This method will create a new experiment with the given `exp_name` if it does not already
        exist in the mlflow server, or it will retrieve the latest run id for the given `exp_name` if
        it does exist. The `run_id` attribute of the `MlFlowLogger` object will be set to the run id
        of the current experiment.

        Typical usage example:
            logger = MlFlowLogger(exp_name='pipeline_name/project_name/your_experiment_name')
            logger._init_experiment()
        """
        experiment = self.client.get_experiment_by_name(self.exp_name)
        if not experiment:
            experiment_id = self.client.create_experiment(self.exp_name)
        else:
            experiment_id = experiment.experiment_id

        run = self.client.create_run(experiment_id)
        self.run_id = run.info.run_id

    def _init_run_id(self) -> None:
        """Initialize the `run_id` attribute by retrieving the latest run id for the current experiment.

        If the `run_id` attribute is already set, it will not be modified. If it is not set, this
        method will retrieve the latest run id for the experiment with the given `exp_name` from the
        mlflow server, and set the `run_id` attribute to this value.

        Typical usage example:
            logger = MlFlowLogger(exp_name='pipeline_name/project_name/your_experiment_name')
            logger._init_run_id()
        """
        if self.run_id:
            pass
        else:
            # Get info about experiment brunch
            exp_info = self.client.get_experiment_by_name(self.exp_name)
            infos = self.client.search_runs([exp_info.experiment_id])
            # Create dict {run_id: start_time}
            ids_list = []
            for i in infos:
                id_date = dict()
                id_date[i.info.run_id] = i.info.start_time
                ids_list.append(id_date)
            # Getting the latest id from sorted list
            sorted_list = sorted(ids_list, key=lambda x: list(x.items())[0][1])
            last_exp = sorted_list[-1]
            self.run_id = list(last_exp.keys())[0]

    def log_experiment_artifacts(
            self,
            config_path: str = None,
            annotation_folder: str = None
    ) -> None:
        """Log experiment artifacts to the mlflow server.

        This method logs the given `config_path` file and all files in the `annotation_folder` to the
        mlflow server as artifacts for the current experiment run. The config will be logged to the
        directory root and annotation to `annotation` directory, respectively. If `config_path` or `annotation_folder`
        is not provided, the corresponding artifacts will not be logged.

        Args:
            config_path: The path to the configuration file to log as an artifact.
            annotation_folder: The path to the folder containing annotation files to log as
                artifacts.

        Typical usage example:
            logger = MlFlowLogger(exp_name='pipeline_name/project_name/your_experiment_name')
            logger.log_experiment_artifacts(
                config_path='path/to/config.yml',
                annotation_folder='path/to/annotation_folder'
            )
        """
        if config_path:
            self.client.log_artifact(self.run_id, config_path)
        if annotation_folder:
            for file_name in os.listdir(annotation_folder):
                if file_name.endswith(".json") or file_name.endswith(".csv"):
                    self.client.log_artifact(
                        self.run_id,
                        os.path.join(annotation_folder, file_name),
                        'annotation'  # Mlflow annotation dir
                    )

    def write_artifact(
            self,
            path2artifact: str,
            path2mlflow: str = None
    ) -> None:
        """Write the given file as an artifact to the current experiment run.

        This method logs the file at the given `filepath` to the mlflow server as an artifact for the
        current experiment run, with the given `artifact_path` as the relative path for the artifact.

        Args:
            path2artifact: The path to the artifact file to log as an artifact.
            path2mlflow: The path to mlflow folder where artifact will be saved

        """
        self.client.log_artifact(
            self.run_id,
            path2artifact,
            path2mlflow
        )

    def register_model(
            self,
            model: nn.Module,
            path2mlflow: str = 'model'
    ) -> None:
        """Save model to local path and register model to the mlflow server.

        This function saves the given `model` to the local path `weights/scripted_model` and registers the
        model to the mlflow server under the current experiment and run. The registered model will be stored
        under the given `path2mlflow` within the experiment run in the mlflow server.

        Args:
            model: The PyTorch model to be saved and registered.
            path2mlflow: The path to store the registered model under within the
                experiment run in the mlflow server. Defaults to 'model'.
        """
        # Save model locally to weights/scripted_model
        local_path = os.path.join(
            os.getcwd(),
            "weights",
            "scripted_model"
        )
        if os.path.exists(local_path):
            shutil.rmtree(local_path)

        print("Saving the model locally...")
        mlflow.pytorch.save_model(model, local_path)
        # Log model to model registry
        print(f"Register to {self.exp_name}/{self.run_id}")
        self.client.log_artifacts(self.run_id, local_path, path2mlflow)

    def write_metrics(
            self,
            metrics: Dict,
            metric_names: List[str],
            step: int
    ) -> None:
        """Write metric values to the mlflow server.

        This function writes the metric values in the given `metrics` dataframe to the mlflow server.

        Args:
            metrics: A dataframe containing the metric values to be logged. The index values
                of the dataframe must correspond to metric names.
            metric_names: A list of strings specifying the columns of the `metrics` dataframe
                to log.
            step: The step at which the metric values are logged.
        """
        # Log filed metrics
        for metric in metric_names:
            value = metrics[metric]
            self.client.log_metric(
                self.run_id,
                metric,
                value,
                step=step
            )

    def write_metric(
            self,
            value: float,
            metric_name: str,
            step: int
    ) -> None:
        """Write a single metric value to the mlflow server.

        This function writes the given `value` for the metric with the given `metric_name` to the mlflow
        server.

        Args:
            value: The value of the metric to be logged.
            metric_name: The name of the metric to be logged.
            step: The step at which the metric value is logged.
        """
        self.client.log_metric(
            self.run_id,
            metric_name,
            value,
            step=step
        )

    def write_loss(
            self,
            value: float,
            split: str,
            step: int,
            name_loss: str = 'loss'
    ) -> None:
        """Write a loss value to the mlflow server.

        This function writes the given `value` for the loss metric with the given `name_loss` to the mlflow
        server under the specified `split` (e.g. train, test).

        Args:
            value: The value of the loss metric to be logged.
            split: The split (e.g. train, test) to log the loss value under.
            step: The step at which the loss value is logged.
            name_loss: The name to give to the loss metric when it is logged. Defaults to 'loss'.
        """
        self.client.log_metric(
            self.run_id,
            f"{split.upper()}/{name_loss}",
            value,
            step=step
        )


if __name__ == '__main__':
    from torchvision.models import MobileNetV2
    model = MobileNetV2()

    exp_name = 'e2e_FieldNet/passport/Tversky_loss_experiment'
    path2myproject = '/Users/anton/PycharmProjects/ml-text_recognition_pipeline'
    mlflow_logger = MlFlowLogger(exp_name)
    mlflow_logger._init_experiment()

    # log_experiment_artifacts - функция не обязательная,
    # можно вообще ничего не логировать в качестве трейн артефактов и не вызывать эту функцию
    # Если вызвали, логирует то что укажете: конфиг и/или csv-шки, json-ы из annotation_folder
    mlflow_logger.log_experiment_artifacts(
        config_path=os.path.join(
            path2myproject,
            'configs/base_config.yml'
        ),
        annotation_folder=os.path.join(
            path2myproject,
            'annotation',
        )
    )

    # Если захочется залогировать кай-нибудь файлик:
    mlflow_logger.write_artifact(
        path2artifact=os.path.join(
            path2myproject,
            'requirements.txt',
        )
    )

    # Если захочется залогировать кай-нибудь файлик в какую-нибудь папочку mlflow:
    mlflow_logger.write_artifact(
        path2artifact=os.path.join(
            path2myproject,
            'requirements.txt',
        ),
        path2mlflow='requirements'
    )

    # При логировании формируется название лосса из split и name_loss f"{split.upper()}/{name_loss}",
    # name_loss по дефолту = "loss"
    # Если нужно логировать разные лоссы, можно переопределить name_loss
    mlflow_logger.write_loss(
        value=999,
        split='train',
        step=1 # Номер эпохи или шага
        #name_loss = FocalLoss
    )

    # Как metric_name называется так она и ляжет в mlflow,
    # если нужно доп разделение можно назвать 'val/F1' или 'test/F1'
    mlflow_logger.write_metric(
        value=0.99,
        metric_name='F1',
        step=1
    )

    # Если нужно залогировать какие-то метрики представленные в DataFrame
    data = [
        [1.0, 1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0, 0.0]
    ]
    metrics = pd.DataFrame(data=data)

    metrics.columns = ['WAR', 'CAR', 'CER', 'WER']
    metrics.index = ['seria', 'number'] # Сформирует название по каждому филду: метрика/филд

    mlflow_logger.write_metrics(
        metrics=metrics,
        metric_names=['WAR', 'CAR'],
        step=1
    )

    # Если нужно залить модель так, чтобы потом можно было добавить ее в реджистри
    mlflow_logger.register_model(model)
    # Модель предварительно сохраняется локально в weights/scripted_model, после чего загружается
    # и логируется в mlflow в дефолтную папку "models"


