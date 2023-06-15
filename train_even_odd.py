import torch.distributed as dist
import pandas as pd
from prettytable import PrettyTable

import torch
import os

from tqdm import tqdm
from omegaconf import DictConfig
from typing import Tuple, List, Dict, Union
from logger import Logger

from utils.tools import get_config, save_json
from train import Trainer


def print_metrics(avg_precision_even, avg_recall_even, avg_f1_score_even,
                  avg_precision_odd, avg_recall_odd, avg_f1_score_odd, common_f1):
    table = PrettyTable()
    table.field_names = ["Metric", "Even", "Odd", "Common"]
    table.add_row(["Precision", f"{avg_precision_even:.6f}", f"{avg_precision_odd:.6f}", ""])
    table.add_row(["Recall", f"{avg_recall_even:.6f}", f"{avg_recall_odd:.6f}", ""])
    table.add_row(["F1 Score", f"{avg_f1_score_even:.6f}", f"{avg_f1_score_odd:.6f}", f"{common_f1:.6f}"])

    print(table)


class MetricsCounter:
    def __init__(self, device):
        self.iterations = 0
        self.cumulative_precision = torch.zeros(size=(1,), dtype=torch.float32).to(device)
        self.cumulative_recall = torch.zeros(size=(1,), dtype=torch.float32).to(device)
        self.cumulative_f1_score = torch.zeros(size=(1,), dtype=torch.float32).to(device)

    def update_metrics(self, precision, recall):
        self.iterations += 1
        self.cumulative_precision += precision
        self.cumulative_recall += recall
        if precision + recall == 0:
            self.cumulative_f1_score += 0
        else:
            f1_score = 2 * (precision * recall) / (precision + recall) + 1e-16
            self.cumulative_f1_score += f1_score

    def get_average_metrics(self):
        if self.iterations == 0:
            return 0, 0, 0
        avg_precision = self.cumulative_precision / self.iterations
        avg_recall = self.cumulative_recall / self.iterations
        avg_f1_score = self.cumulative_f1_score / self.iterations
        return avg_precision.item(), avg_recall.item(), avg_f1_score.item()


class TrainerEvenOdd(Trainer):
    """
    Base class for training deep learning models.

    This class handles the training process of a model, including data_text_recognition loading, model
    initialization, forward and backward steps, loss calculation, optimization,
    scheduling, and evaluation.

    Args:
        config: A dictionary-like object containing the configuration
            parameters for the training process.

    Attributes:
        config (DictConfig): The configuration parameters for the training process.
        logger (Logger): An instance of the `Logger` class, used to log training
            and evaluation metrics to TensorBoard and MlFlow.
        device (torch.device): The device to be used for training.
        model (nn.Module): The model to be trained.
        criterion (callable): The loss function to be used during training.
        optimizer (torch.optim.Optimizer): The optimizer to be used to update the
            model parameters during training.
        scheduler (callable): The scheduler to be used to adjust the learning rate
            during training.
        scaler (torch.cuda.amp.GradScaler): An instance of the `GradScaler` class,
            used to scale gradients during mixed precision training.
        train_loader (torch.utils.data_text_recognition.DataLoader): A PyTorch DataLoader for the
            training dataset.
        val_loader (torch.utils.data_text_recognition.DataLoader): A PyTorch DataLoader for the
            validation dataset.
        best_war (float): The best weighted average word accuracy rate achieved during
            training.
        best_car (float): The best char accuracy rate achieved during
            training.
        save_best_folder (str): The path to the folder where the best model
            checkpoints will be saved.
        best_checkpoints (list): A list of tuples containing the epoch and score
            of the best model checkpoints saved during training.
        current_epoch (int): The current epoch during training.

    Typical usage example:
        from utils.tools import get_config

        path2config = path/to/config
        config = get_config(path2config)
        trainer = Trainer(config)
        trainer.train()
    """
    def __init__(
            self,
            config: DictConfig
    ):
        super(TrainerEvenOdd, self).__init__(config)

    def train_step(self) -> None:
        """
        Perform a training step over the training dataset.

        This method sets the model in train mode, iterates over the training
        data_text_recognition, performs a forward step, calculates the loss, and performs
        backpropagation and optimization. It also logs the batch loss and
        updates the learning rate using the scheduler.
        """

        # Set train mode
        self.model.train()

        # Set train loss counter
        train_loss = torch.scalar_tensor(0)
        even_loss_log = torch.scalar_tensor(0)
        odd_loss_log = torch.scalar_tensor(0)
        counter = torch.scalar_tensor(0)

        loop = tqdm(self.train_loader, desc='Training', disable=not self.config["main_process"])

        # Iter done
        iter_done = len(self.train_loader) * self.current_epoch

        # Iter by loop
        for i, (batch) in enumerate(loop):
            iter_done += 1
            counter += 1

            # Forward step
            preds, loss_inp = self._forward_step(self.model, batch)
            even_out, odd_out = preds

            # Calculate loss
            even_loss = self.criterion(even_out, loss_inp['even_inp'])
            odd_loss = self.criterion(odd_out, loss_inp['odd_inp'])
            full_loss = even_loss['loss'] + odd_loss['loss']

            # Backward
            self.scaler.scale(full_loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad(set_to_none=True)

            # Update loss
            batch_loss = full_loss.detach().cpu().item()
            train_loss += batch_loss

            batch_even_loss_log = even_loss['loss'].detach().cpu().item()
            even_loss_log += batch_even_loss_log

            batch_odd_loss_log = odd_loss['loss'].detach().cpu().item()
            odd_loss_log += batch_odd_loss_log

            # Log batch loss
            self.logger.log_batch_loss(batch_loss, 'train', iter_done)
            self.logger.log_batch_loss(batch_even_loss_log, 'train_even_loss', iter_done)
            self.logger.log_batch_loss(batch_odd_loss_log, 'train_odd_loss', iter_done)

            # Set loss value for loop
            loop.set_postfix({
                "loss": float(train_loss / (i + 1)),
                "loss_even": float(even_loss_log / (i + 1)),
                "loss_odd": float(odd_loss_log / (i + 1))
            })

        self.scheduler.step()
        if self.config["train"]["use_ddp"]:
            counter = counter.to(self.device)
            train_loss = train_loss.to(self.device)
            dist.all_reduce(counter, op=dist.ReduceOp.SUM)
            dist.all_reduce(train_loss, op=dist.ReduceOp.SUM)

        # Print mean train loss
        train_loss = train_loss / counter
        if self.config['main_process']:
            print(f"train loss - {train_loss}")

        # Log epoch loss to Tensorboard and MlFlow
        self.logger.log_epoch_loss(
            train_loss.detach().cpu().item(),
            'train',
            self.current_epoch
        )

        self.logger.log_lr(
            self.optimizer.param_groups[0]['lr'],
            self.current_epoch
        )

    def evaluate(
            self,
            model: torch.nn.Module,
            loader
    ) -> None:
        """
        Evaluate the model on the given data_text_recognition loader and model.

        This method sets the model in eval mode, iterates over the data_text_recognition in the
        given `loader`, performs a forward step, calculates the loss, and
        translates the predicted and target sequences to text. It also logs the
        epoch loss to TensorBoard and MlFlow.

        Args:
            model: Evaluating model
            loader: A data_text_recognition loader that yields batches of data_text_recognition to be evaluated.

        Returns:
            A tuple containing three lists:
            - A list of target texts.
            - A list of predicted texts.
            - A list of metadata strings.
        """
        # Set model to eval mode
        model.eval()

        # Set eval loss counter
        self.eval_loss = torch.scalar_tensor(0)
        eval_loss = torch.scalar_tensor(0)
        even_loss_log = torch.scalar_tensor(0)
        odd_loss_log = torch.scalar_tensor(0)

        loop = tqdm(loader, desc='Evaluate')

        self.even_metrics_counter = MetricsCounter(device=self.device)
        self.odd_metrics_counter = MetricsCounter(device=self.device)

        with torch.no_grad():
            # Iter by loop
            for i, (batch) in enumerate(loop):

                # Forward step
                preds, loss_inp = self._forward_step(model, batch)
                even_out, odd_out = preds

                # Calculate loss
                even_loss = self.criterion(even_out, loss_inp['even_inp'])
                odd_loss = self.criterion(odd_out, loss_inp['odd_inp'])

                # Calculate loss
                full_loss = even_loss['loss'] + odd_loss['loss']

                # Update loss
                batch_loss = full_loss.detach().cpu().item()
                self.eval_loss += batch_loss

                batch_even_loss_log = even_loss['loss'].detach().cpu().item()
                even_loss_log += batch_even_loss_log

                batch_odd_loss_log = odd_loss['loss'].detach().cpu().item()
                odd_loss_log += batch_odd_loss_log

                # Log batch loss
                # Set loss value for loop
                loop.set_postfix({
                    "loss": float(self.eval_loss / (i + 1)),
                    "loss_even": float(even_loss_log / (i + 1)),
                    "loss_odd": float(odd_loss_log / (i + 1))
                })

                # Calculate metrics
                precision_even, recall_even = self.calculate_metrics(
                    even_out[:, 0, :, :], loss_inp['even_inp']['shrink_map']
                )
                precision_odd, recall_odd = self.calculate_metrics(
                    odd_out[:, 0, :, :], loss_inp['odd_inp']['shrink_map']
                )

                self.even_metrics_counter.update_metrics(precision_even, recall_even)
                self.odd_metrics_counter.update_metrics(precision_odd, recall_odd)

            # Print mean train loss
            self.eval_loss = self.eval_loss / len(loader)
            even_loss_log = even_loss_log / len(loader)
            odd_loss_log = odd_loss_log / len(loader)


        # Log epoch loss to Tensorboard and MlFlow
        self.logger.log_epoch_loss(
            self.eval_loss.item(),
            'val',
            self.current_epoch
        )

        self.logger.log_epoch_loss(
            even_loss_log.item(),
            'val_even',
            self.current_epoch
        )

        self.logger.log_epoch_loss(
            odd_loss_log.item(),
            'val_odd',
            self.current_epoch
        )

    def _forward_step(
            self,
            model: torch.nn.Module,
            batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ) -> Tuple[List[Union[torch.Tensor, torch.Tensor]], Dict]:
        """
        Perform a forward step for a given batch of data.

        This method moves the data to the correct device, performs a forward
        step using the model, and returns the predicted sequences for even and odd targets,
        as well as the loss inputs for the batch.

        Args:
            model (torch.nn.Module): The learning model.
            batch (Tuple[torch.Tensor, torch.Tensor, torch.Tensor]): A tuple containing three tensors:
                - A tensor of images (torch.Tensor): Shape (batch_size, num_channels, height, width).
                - A tensor of even_targets (torch.Tensor): Shape (batch_size, 4, height, width).
                - A tensor of odd_targets (torch.Tensor): Shape (batch_size, 4, height, width).

        Returns:
            Tuple[List[torch.Tensor, torch.Tensor], Dict]: A tuple containing two elements:
                - A list containing two tensors:
                    - The predicted sequence tensor for even targets (torch.Tensor): Shape (batch_size, 4, height, width).
                    - The predicted sequence tensor for odd targets (torch.Tensor): Shape (batch_size, 4, height, width).
                - A dictionary containing the loss inputs for the batch (Dict):
                    - "even_inp" (Dict): A dictionary containing the following keys and tensors for even targets:
                        - "shrink_map" (torch.Tensor): Shape (batch_size, height, width).
                        - "shrink_mask" (torch.Tensor): Shape (batch_size, height, width).
                        - "threshold_map" (torch.Tensor): Shape (batch_size, height, width).
                        - "threshold_mask" (torch.Tensor): Shape (batch_size, height, width).
                    - "odd_inp" (Dict): A dictionary containing the following keys and tensors for odd targets:
                        - "shrink_map" (torch.Tensor): Shape (batch_size, height, width).
                        - "shrink_mask" (torch.Tensor): Shape (batch_size, height, width).
                        - "threshold_map" (torch.Tensor): Shape (batch_size, height, width).
                        - "threshold_mask" (torch.Tensor): Shape (batch_size, height, width).
        """
        image_batch, even_targets, odd_targets = batch

        image_batch = image_batch.to(self.device)
        image_batch.div_(255.)
        image_batch.sub_(self.mean)
        image_batch.div_(self.std)

        even_targets = even_targets.to(self.device)
        odd_targets = odd_targets.to(self.device)

        out_even, out_odd = model(image_batch)

        loss_inputs = {}
        for name, target in zip(['even_inp', 'odd_inp'], [even_targets, odd_targets]):
            loss_input = {
                'shrink_map': target[:, 0, :, :],
                'shrink_mask': target[:, 1, :, :],
                'threshold_map': target[:, 2, :, :],
                'threshold_mask': target[:, 3, :, :]
            }
            loss_inputs[name] = loss_input

        return [out_even, out_odd], loss_inputs

    def check_current_result(
        self,
        model: torch.nn.Module,
        metrics: pd.DataFrame
    ) -> None:
        """Check the current result of the model and save the best model checkpoint if necessary.

        Args:
            model: Epoch-trained model
            metrics: DataFrame containing the metrics for the current epoch.
        """
        # TODO: Кажется, что нужен более гибкий критерий выбора метрики для сохранения ckpt

        avg_precision_even, avg_recall_even, avg_f1_score_even = self.even_metrics_counter.get_average_metrics()
        avg_precision_odd, avg_recall_odd, avg_f1_score_odd = self.odd_metrics_counter.get_average_metrics()

        common_f1 = (avg_f1_score_even + avg_f1_score_odd)/2

        print_metrics(
            avg_precision_even, avg_recall_even, avg_f1_score_even,
            avg_precision_odd, avg_recall_odd, avg_f1_score_odd, common_f1
        )

        dict_for_logging = {
            'avg_precision_even': avg_precision_even,
            'avg_recall_even': avg_recall_even,
            'avg_f1_score_even': avg_f1_score_even,
            'avg_precision_odd': avg_precision_odd,
            'avg_recall_odd': avg_recall_odd,
            'avg_f1_score_odd': avg_f1_score_odd,
            'common_f1': common_f1
        }

        self.logger.log_epoch_metrics(dict_for_logging, dict_for_logging.keys(), self.current_epoch)
        # Check both metrics
        if common_f1 > self.best_score:
            print("Saving best model ...")
            self.best_score = common_f1
            # Create path to best result
            path2best_weight = os.path.join(
                self.save_best_folder,
                f"{self.config['description']['experiment_name']}_score_{self.best_score}.pth"
            )

            # Save model
            torch.save(model.state_dict(), path2best_weight)
            # Append current best checkpoint path to the list
            self.best_checkpoints.append(path2best_weight)

            # Logic for holding best K checkpoint
            if len(self.best_checkpoints) > self.config["checkpoint"]['average_top_k']:
                self.best_checkpoints.pop(0)

            # Save list with paths to best checkpoints
            path2best_checkpoints = os.path.join(self.save_best_folder, 'top_checkpoints.json')

            save_json(
                self.best_checkpoints,
                path2best_checkpoints
            )


if __name__ == '__main__':
    config = get_config()
    trainer = TrainerEvenOdd(config)
    trainer.train()