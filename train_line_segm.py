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

from utils.tools import get_config, save_json
from utils.convert_weights import tracing_weights, average_weights
from utils.metrics import draw_polygons

from criterions import get_criterion
from train import Trainer
import albumentations as A

from time import time

class TrainerLD(Trainer):
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
        super(TrainerLD, self).__init__(config)

        self.criterion = get_criterion(config["loss_function"]).to(self.device)
        self.scaler = torch.cuda.amp.GradScaler()

        print(self.criterion)

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
        if config['model']['tune_model']:
            if config['model']['freeze_all']:
                print('freeze backbone and neck')
                self.model.freeze_feature_extractor()
            elif config['model']['freeze_backbone']:
                print('freeze only backbone')
                self.model.freeze_feature_backbone()

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
            out, loss_inp = self._forward_step(self.model, batch)
            if i == 0:
                print(out.shape)

            # Calculate loss
            print(out.shape, loss_inp.shape)
            full_loss = self.criterion(out, loss_inp)
            # Backward
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

    def evaluate_step(self, model: torch.nn.Module) -> None:
        """Evaluates the model's performance on the val dataset.

        This method gets the predicted values for the validation data_text_recognition, calculates
        evaluation metrics, and logs the epoch metrics to TensorBoard and MlFlow.
        If the metrics improved, the current model weights are saved as a checkpoint.
        Metrics also will be printed.

        Args:
            model: Evaluating model
        """
        # Get predicted values
        self.evaluate(model, self.val_loader)

        # Save weights if metrics were improved
        self.check_current_result(model, self.eval_loss)

        # Print metrics for current epoch
        self.evaluator.clear_data()

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

        self.precision = torch.zeros(size=(1,), dtype=torch.float32).to(self.device)
        self.recall = torch.zeros(size=(1,), dtype=torch.float32).to(self.device)
        self.f1_score = torch.zeros(size=(1,), dtype=torch.float32).to(self.device)

        with torch.no_grad():
            # Iter by loop
            for i, (batch) in enumerate(loop):

                # Forward step
                out, loss_inp = self._forward_step(self.model, batch)
                if i == 0:
                    print(out.shape)
                # Calculate metrics
                precision, recall = self.calculate_metrics(out[:, 0, :, :], loss_inp[:, 0, :, :])

                # Calculate loss
                full_loss = self.criterion(out, loss_inp)

                # Update loss
                self.eval_loss += full_loss.detach().cpu().item()

                # Set loss value for loop
                loop.set_postfix({"loss": float(self.eval_loss / (i+1))})

                self.precision += precision
                self.recall += recall

            self.precision /= (i + 1)
            self.recall /= (i + 1)
            self.f1_score += 2 * self.precision * self.recall / (self.precision + self.recall) + 1e-16

            # Print mean train loss
            self.eval_loss = self.eval_loss / len(loader)
            print(f"evaluate loss - {self.eval_loss}")
            print(f"PRECISION - {self.precision}")
            print(f"RECALL - {self.recall}")
            print(f"F1_SCORE - {self.f1_score}")

        # Log epoch loss to Tensorboard and MlFlow
        self.logger.log_epoch_loss(
            self.eval_loss.item(),
            'val',
            self.current_epoch
        )

    def _forward_step(
            self,
            model: torch.nn.Module,
            batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, List]
    ) -> Tuple[List[torch.Tensor], List[Dict]]:
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
        image_batch, masks = batch

        image_batch = image_batch.to(self.device)
        masks = masks.to(self.device)

        image_batch.div_(255.)
        image_batch.sub_(self.mean)
        image_batch.div_(self.std)

        out = model(image_batch)

        return out, masks


if __name__ == "__main__":

    config = get_config()
    trainer = TrainerLD(config)
    trainer.train()