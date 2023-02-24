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


class TrainerV2(Trainer):
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
        super(TrainerV2, self).__init__(config)

        self.criterion = get_criterion(config["loss_function"]).to(self.device)

        self.resize = A.LongestMaxSize(max_size=1024)

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
            out, loss_inp = self._forward_step(self.model, batch)
            # Calculate loss
            full_loss = self.criterion(out, loss_inp)
            # Backward
            full_loss['loss'].backward()
            self.optimizer.step()
            # self.scaler.scale(loss).backward()
            # self.scaler.step(self.optimizer)
            # self.scaler.update()
            self.optimizer.zero_grad(set_to_none=True)

            # Update loss
            batch_loss = full_loss['loss'].detach().cpu().item()
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

        with torch.no_grad():
            # Iter by loop
            for i, (batch) in enumerate(loop):

                # Forward step
                out, loss_inp = self._forward_step(self.model, batch)

                # Calculate loss
                full_loss = self.criterion(out, loss_inp)

                # Update loss
                self.eval_loss += full_loss['loss'].detach().cpu().item()

                # Set loss value for loop
                loop.set_postfix({"loss": float(self.eval_loss / (i+1))})

            # Print mean train loss
            self.eval_loss = self.eval_loss / len(loader)
            print(f"evaluate loss - {self.eval_loss}")

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
        image_batch, targets = batch

        targets_even, targets_odd = targets

        shrink_maps_batch_even = targets_even[0].to(self.device)
        shrink_masks_batch_even = targets_even[1].to(self.device)
        threshold_maps_batch_even = targets_even[2].to(self.device)
        threshold_masks_batch_even = targets_even[3].to(self.device)

        shrink_maps_batch_odd = targets_odd[0].to(self.device)
        shrink_masks_batch_odd = targets_odd[1].to(self.device)
        threshold_maps_batch_odd = targets_odd[2].to(self.device)
        threshold_masks_batch_odd = targets_odd[3].to(self.device)

        image_batch = image_batch.to(self.device)

        image_batch.div_(255.)
        image_batch.sub_(self.mean)
        image_batch.div_(self.std)

        out = model(image_batch)

        loss_inp_even = {
            'shrink_map': shrink_maps_batch_even,
            'shrink_mask': shrink_masks_batch_even,
            'threshold_map': threshold_maps_batch_even,
            'threshold_mask': threshold_masks_batch_even
        }

        loss_inp_odd = {
            'shrink_map': shrink_maps_batch_odd,
            'shrink_mask': shrink_masks_batch_odd,
            'threshold_map': threshold_maps_batch_odd,
            'threshold_mask': threshold_masks_batch_odd
        }

        return out, [loss_inp_even, loss_inp_odd]

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
        if self.eval_loss <= self.best_score:
            print("Saving best model ...")
            self.best_score = self.eval_loss

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

    def convert_weights(self) -> None:
        """Average the weights of the best K models and trace the model for serving.

       This method averages the weights of the best K models based on the WAR and CAR
       metrics, and then traces the model using the PyTorch JIT library. It also logs
       the traced model to MlFlow.
       """

        self.config["train"]["use_ddp"] = False
        self.model = average_weights(self.config, self.best_checkpoints)
        tracing_weights(self.model, self.config)
        self.logger.log_model(model=self.model, path2mlflow='traced_model')
        self.logger.log_model(local_model_path=self.best_checkpoints[-1], path2mlflow='best_weight')

    def resize_for_tensorboard(self, batch_images):
        return F.interpolate(batch_images, size=(self.config.logging.image.width, self.config.logging.image.height),
                             mode='bilinear', align_corners=False).cpu().numpy()

    def plot_result(self, mode, batch_idx, batch, preds):
        imgs, shrink_maps, shrink_masks, threshold_maps, threshold_masks, gt_polygons = batch
        self.logger.tensor_log.writer.add_images(f"{mode}_gt/original",
                                          self.resize_for_tensorboard(imgs.detach().sigmoid()),
                                          self.current_epoch + batch_idx)
        self.logger.tensor_log.writer.add_images(f"{mode}_gt/shrink_maps",
                                          self.resize_for_tensorboard(shrink_maps.unsqueeze(1).detach()),
                                          self.current_epoch + batch_idx)

        self.logger.tensor_log.writer.add_images(f"{mode}_gt/shrink_masks",
                                          self.resize_for_tensorboard(shrink_masks.unsqueeze(1).detach()),
                                          self.current_epoch + batch_idx)
        self.logger.tensor_log.writer.add_images(f"{mode}_gt/threshold_maps",
                                          self.resize_for_tensorboard(threshold_maps.unsqueeze(1).detach()),
                                          self.current_epoch + batch_idx)
        self.logger.tensor_log.writer.add_images(f"{mode}_gt/threshold_masks",
                                          self.resize_for_tensorboard(threshold_masks.unsqueeze(1).detach()),
                                          self.current_epoch + batch_idx)
        self.logger.tensor_log.writer.add_images(f"{mode}_preds/shrink_maps",
                                          self.resize_for_tensorboard(preds[:, [0], :, :].detach()),
                                          self.current_epoch + batch_idx)
        self.logger.tensor_log.writer.add_images(f"{mode}_preds/threshold_maps",
                                          self.resize_for_tensorboard(preds[:, [1], :, :].detach()),
                                          self.current_epoch + batch_idx)
        self.logger.tensor_log.writer.add_images(f"{mode}_preds/binary_maps",
                                          self.resize_for_tensorboard(preds[:, [2], :, :].detach()),
                                          self.current_epoch + batch_idx)

    def plot_bbox_val(self, stage, imgs, polygons, batch_idx):
        contour_plots = []
        for img, pol in zip(imgs.detach().sigmoid().cpu().numpy(), polygons):
            img = np.moveaxis(img, 0, -1)
            img = np.round(img * 255)
            img = draw_polygons(img, pol, contours=True)
            img = cv2.resize(img, (self.config.logging.image.width, self.config.logging.image.height))
            img = np.moveaxis(img, -1, 0)
            img = np.clip(img / 255, 0, 1)
            contour_plots.append(img)
        contour_plots = np.array(contour_plots)
        self.logger.tensor_log.writer.add_images(f"val_{stage}/contours", contour_plots, self.current_epoch + batch_idx)


if __name__ == "__main__":

    config = get_config()
    trainer = TrainerV2(config)
    trainer.train()