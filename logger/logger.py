import os

from tensorboardX import SummaryWriter
from clearml import OutputModel
from clearml.task import TaskInstance


class Logger:
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
            log_dir: str,
            task: TaskInstance = None,
            n_debug_samples: int = 100,
    ):
        self.writer = SummaryWriter(log_dir) if log_dir else None
        if task:
            self.task = task
            self.clearml_logger = self.task.get_logger().set_default_debug_sample_history(n_debug_samples)
            self.output_model = OutputModel(task=task, framework='PyTorch')

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
        if self.writer:
            self.writer.add_scalar(name, value, step)
            self.writer.flush()

    def report_plot(self, plt, title: str, series: str, iteration: int):
        if self.task:
            self.task.get_logger().report_matplotlib_figure(
                title=title,
                series=series,
                iteration=iteration,
                figure=plt,
                report_image=True,
        )

    def upload_artifact(self, name, path2artifact):
        if self.task:
            self.task.upload_artifact(
                name,
                artifact_object=os.path.join(
                    path2artifact
                )
            )

    def report_plotly(self, plt, title: str, series: str, iteration: int):
        if self.task:
            self.task.get_logger().report_plotly(
                title=title, series=series, iteration=iteration, figure=plt
            )

    def report_image(self, image, title, series, iteration):
        if self.task:
            self.task.get_logger().report_image(
                title,
                series,
                iteration=iteration,
                image=image
        )

    def report_text(self, text):
        if self.task:
            self.task.get_logger().report_text(text)

    def log_model(self, weights_filename):
        if self.task:
            self.output_model.update_weights(weights_filename)

    def write_metrics(
            self,
            metrics: dict,
            step: int,
    ) -> None:

        if self.writer:
            for metric_name, items in metrics.items():
                for entitie_name, entitie_value in items.items():
                    name_metric = f'{metric_name}/{entitie_name}'
                    self.writer.add_scalar(name_metric, entitie_value, step)
            self.writer.flush()


if __name__ == "__main__":
    import sys
    # Эт чтобы нормально get_config импортнулся
    sys.path.append('/home/addudkin/ml-text_recognition_pipeline')
    import torch
    import numpy as np
    import matplotlib.pyplot as plt
    import plotly.express as px

    from utils.tools import set_random_seed, get_config
    from PIL import Image
    from torchvision.models import resnet50

    set_random_seed()
    config, task = get_config()
    logger = Logger(config['tensorboard']['log_dir'], task)

    logger.report_text('Hello, logging test has started')

    x = 0.0
    for i in range(0, 10):
        print('log scalar', x)
        logger.log_scalar(x, 'loss', i)
        x += 0.5
        x = x**2

    ### MATPLOTLIB BLOCK ###
    for iteration in range(10):
        N = 50
        x = np.random.rand(N)
        y = np.random.rand(N)
        colors = np.random.rand(N)
        area = (30 * np.random.rand(N)) ** 2  # 0 to 15 point radii
        plt.scatter(x, y, s=area, c=colors, alpha=0.5)
        logger.report_plot(plt=plt, title=f'Manual Reporting', series="Just a plot", iteration=iteration)
    ### MATPLOTLIB BLOCK ###

    logger.upload_artifact(name='Config of model', path2artifact=config['config_path'])

    ### PLOTLY BLOCK ###
    # Iris dataset
    df = px.data.iris()
    # create complex plotly figure
    fig = px.scatter(df, x="sepal_width", y="sepal_length", color="species", marginal_y="rug", marginal_x="histogram")
    # report the plotly figure
    for iteration in range(10):
        logger.report_plotly(title=f"iris", series="sepal", iteration=iteration, plt=fig)
    ### PLOTLI BLOCK ###

    for iteration in range(10):
        image = Image.open('test_image/test_image.jpeg')
        logger.report_image(image=image, title="Test image", series=f"Keanu_{iteration}", iteration=iteration)

    model = resnet50(weights="IMAGENET1K_V2")
    torch.save(model.state_dict(), 'test_image/weights.pth')
    logger.log_model('test_image/weights.pth')

    logger.write_metrics(
        {
            'F1_score': {'cats': 0.9, 'dogs': 0.6}
        },
        step=0
    )

    logger.report_text('Goodbye, logging test has finished')
