import torch.distributed as dist
import pandas as pd
import numpy as np
import torch.nn.functional as F

import torch
import cv2
import os

from tqdm import tqdm
from omegaconf import DictConfig
from typing import Tuple, List, Dict
from logger import Logger

from utils.tools import get_config, get_device, save_json
from utils.convert_weights import tracing_weights, average_weights
from utils.metrics import show_metrics, Evaluator, SegPostprocessing, draw_polygons

from criterions import get_criterion
from optimizers import get_optimizer_and_scheduler
from datasets import get_dataset, get_dataloader
from models import get_model

from train_det_and_lines import TrainerTDLine


class TrainerTDLineEval(TrainerTDLine):
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
        super(TrainerTDLineEval, self).__init__(config)

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
        self.eval_loss = torch.scalar_tensor(0).cpu()

        loop = tqdm(loader, desc='Evaluate')

        self.precision_td = torch.zeros(size=(1,), dtype=torch.float32).to(self.device)
        self.recall_td = torch.zeros(size=(1,), dtype=torch.float32).to(self.device)
        self.f1_score_td = torch.zeros(size=(1,), dtype=torch.float32).to(self.device)

        self.precision_line = torch.zeros(size=(1,), dtype=torch.float32).to(self.device)
        self.recall_line = torch.zeros(size=(1,), dtype=torch.float32).to(self.device)
        self.f1_score_line = torch.zeros(size=(1,), dtype=torch.float32).to(self.device)

        self.f1_score = torch.zeros(size=(1,), dtype=torch.float32).to(self.device)

        with torch.no_grad():
            # Iter by loop
            for i, (batch) in enumerate(loop):

                # Forward step
                preds, loss_input_td, loss_input_lines = self._forward_step(self.model, batch)

                # Calculate loss
                td_loss = self.criterion_td(preds[:, :3, :, :], loss_input_td)

                line_pred_inp = preds[:, 3:, :, :].contiguous()
                lines_loss = self.criterion_lines(line_pred_inp, loss_input_lines)

                td_loss = td_loss['loss']
                full_loss = td_loss + lines_loss

                # Calculate metrics for text_det
                precision_td, recall_td = self.calculate_metrics(preds[:, 0, :, :], loss_input_td['shrink_map'])
                precision_line, recall_line = self.calculate_metrics(preds[:, 3, :, :], loss_input_lines[:, 0, :, :])

                # Update loss
                self.eval_loss += full_loss.detach().cpu().item()

                # Set loss value for loop
                loop.set_postfix({"loss": float(self.eval_loss / (i+1))})

                self.precision_td += precision_td
                self.precision_line += precision_line

                self.recall_td += recall_td
                self.recall_line += recall_line

            self.precision_td /= (i + 1)
            self.precision_line /= (i + 1)

            self.recall_td /= (i + 1)
            self.recall_line /= (i + 1)

            self.f1_score_td += 2 * self.precision_td * self.recall_td / (self.precision_td + self.recall_td) + 1e-16
            self.f1_score_line += 2 * self.precision_line * self.recall_line / (self.precision_line + self.recall_line) + 1e-16

            # Print mean train loss
            self.eval_loss = self.eval_loss / len(loader)
            print(f"evaluate loss - {self.eval_loss}")
            print(f"precision_td - {self.precision_td}")
            print(f"recall_td - {self.recall_td}")
            print(f"f1_td - {self.f1_score_td}")

            print(f"precision_line - {self.precision_line}")
            print(f"recall_line - {self.recall_line}")
            print(f"f1_line - {self.f1_score_line}")

            self.f1_score = (self.f1_score_line + self.f1_score_td)/2
            print(f"mean_f1 - {self.f1_score}")
        # Log epoch loss to Tensorboard and MlFlow
        self.logger.log_epoch_loss(
            self.eval_loss.item(),
            'val',
            self.current_epoch
        )


if __name__ == "__main__":
    config = get_config()
    trainer = TrainerTDLineEval(config)
    trainer.evaluate(trainer.model, trainer.val_loader)