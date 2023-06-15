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

from train import Trainer


class TrainerTDLine(Trainer):
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

        self.config = config
        # Init Logger
        self.logger = Logger(self.config)

        # Init base train element
        self.device = get_device(config)
        self.model = get_model(config, self.device).to(self.device)
        self.optimizer, self.scheduler = get_optimizer_and_scheduler(self.model, config)

        self.mean = torch.FloatTensor(config.data.mean).view(-1, 1, 1).to(self.device)
        self.std = torch.FloatTensor(config.data.std).view(-1, 1, 1).to(self.device)

        # Init datasets and loaders
        train_dataset = get_dataset("train", config)
        val_dataset = get_dataset("val", config)

        self.train_loader = get_dataloader(train_dataset, config)
        self.val_loader = get_dataloader(val_dataset, config)

        self.scaler = torch.cuda.amp.GradScaler()

        # Set other parameters
        self.save_best_folder = os.path.join(
            os.getcwd(),
            "weights",
            config["description"]["project_name"],
            config["description"]["experiment_name"]
        )
        os.makedirs(self.save_best_folder, exist_ok=True)
        self.best_checkpoints = list()

        # Set metric
        self.best_score = 0
        # Set Postprocessor
        self.criterion_td = get_criterion(config["loss_function"]['classic_DB']).to(self.device)
        self.criterion_lines = get_criterion(config["loss_function"]['lines_segm']).to(self.device)
        print(self.criterion_lines)

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
        counter = torch.scalar_tensor(0)

        loop = tqdm(self.train_loader, desc='Training', disable=not self.config["main_process"])

        # Iter done
        iter_done = len(self.train_loader) * self.current_epoch

        # Iter by loop
        for i, (batch) in enumerate(loop):
            iter_done += 1
            counter += 1

            # Forward step
            preds, loss_input_td, loss_input_lines = self._forward_step(self.model, batch)

            td_loss = self.criterion_td(preds[:, :3, :, :], loss_input_td)

            line_pred_inp = preds[:, 3:, :, :].contiguous()
            lines_loss = self.criterion_lines(line_pred_inp, loss_input_lines)

            # Backward
            td_loss = td_loss['loss']
            full_loss = td_loss + lines_loss

            self.scaler.scale(full_loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad(set_to_none=True)

            # Update loss
            batch_loss = full_loss.detach().cpu().item()
            train_loss += batch_loss

            # Log batch loss
            self.logger.log_batch_loss(batch_loss, 'train', iter_done)

            # Set loss value for loop
            loop.set_postfix({"loss": float(train_loss / (i+1))})

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

    def _forward_step(
            self,
            model: torch.nn.Module,
            batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict, Dict]:
        """
        Perform a forward step for a given batch of data_text_recognition.

        This method moves the data_text_recognition to the correct device, performs a forward
        step using the model, and returns the predicted and target sequences,
        as well as the length of the predicted and target sequences, and the
        metadata for the batch.

        Args:
            model: Our learning  model
            batch: A tuple containing four tensors:
                - A tensor of images.
                - A tensor of targets.
                - A tensor of target lengths.
                - A tensor of metadata.

        Returns:
            A tuple containing five tensors:
            - The predicted sequence tensor.
            - The target sequence tensor.
            - The tensor with lengths of predicted tensors
            - The tensor with lengths of target tensors
            - The labels sequence tensor
        """
        image_batch, shrink_maps_batch, shrink_masks_batch,\
        threshold_maps_batch, threshold_masks_batch, lines_mask_batch = batch

        image_batch = image_batch.to(self.device)
        image_batch.div_(255.)
        image_batch.sub_(self.mean)
        image_batch.div_(self.std)

        shrink_maps_batch = shrink_maps_batch.to(self.device)
        shrink_masks_batch = shrink_masks_batch.to(self.device)
        threshold_maps_batch = threshold_maps_batch.to(self.device)
        threshold_masks_batch = threshold_masks_batch.to(self.device)

        lines_mask_batch = lines_mask_batch.to(self.device)

        out = model(image_batch)

        loss_input_td = {
            'shrink_map': shrink_maps_batch,
            'shrink_mask': shrink_masks_batch,
            'threshold_map': threshold_maps_batch,
            'threshold_mask': threshold_masks_batch
        }

        return out, loss_input_td, lines_mask_batch

    def check_current_result(
            self,
            model: torch.nn.Module,
            metrics: pd.DataFrame) -> None:
        """Check the current result of the model and save the best model checkpoint if necessary.

        Args:
            model: Epoch-trained model
            metrics: DataFrame containing the metrics for the current epoch.
        """
        # TODO: Кажется, что нужен более гибкий критерий выбора метрики для сохранения ckpt

        # Check both metrics
        if self.f1_score > self.best_score:
            print("Saving best model ...")
            self.best_score = self.f1_score.detach().cpu().item()
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


    def calculate_metrics(self, batch_pred, batch_target, thr=0.5):
        batch_pred = (batch_pred > thr).float() * 1
        precision = 0
        recall = 0

        for pred, target in zip(batch_pred, batch_target):
            tp = target * pred
            n_gt = torch.count_nonzero(target)
            if not n_gt:
                continue

            n_tp = torch.count_nonzero(tp)
            n_prd = torch.count_nonzero(pred)

            precision += n_tp / n_prd if n_prd != 0 else 0
            recall += n_tp / n_gt if n_gt != 0 else 0

        return precision/batch_pred.shape[0], recall/batch_pred.shape[0]


if __name__ == "__main__":
    config = get_config()
    trainer = TrainerTDLine(config)
    trainer.train()