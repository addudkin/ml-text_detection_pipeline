import numpy as np
import torch.distributed as dist
import pandas as pd
import torch
import os

from tqdm import tqdm
from omegaconf import DictConfig
from typing import Tuple, List, Dict
from logger import Logger
from utils.hmean import HmeanIOUMetric
from utils.metrics import FullTextPostProcessor

from utils.tools import get_device
from utils.convert_weights import tracing_weights, average_weights
from criterions import get_criterion
from optimizers import get_optimizer_and_scheduler
from datasets import get_dataset, get_dataloader
from models import get_model
from utils.types import ImageResizerResult, SegmentationMaskResult


class Trainer:
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
        self.criterion = get_criterion(config["loss_function"]).to(self.device)
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
        # Set Evaluator
        self.postprocessor = FullTextPostProcessor(**config['post_processor']['params'])

    def train(self) -> None:
        """
        Train the model for a number of epochs.

        This method iterates over the training epochs and includes:
        - training steps
        - evaluating steps with saving weights and showing metrics
        - convert and average weights
        """
        # Iter by epoch
        for epoch in range(self.config["train"]["epoch"]):
            self.current_epoch = epoch

            if self.config["main_process"]:
                print(f"\nCurrent learning rate is: {self.optimizer.param_groups[0]['lr']:.6f}")

            self.train_step()

            if self.config['train']['use_ddp']:
                if self.config["main_process"]:
                    self.evaluate_step(self.model.module)
                torch.distributed.barrier()
            else:
                self.evaluate_step(self.model)

        if self.config["main_process"]:
            self.convert_weights()

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
            self.optimizer.zero_grad(set_to_none=True)

            # Forward step
            out, loss_inp = self._forward_step(self.model, batch)

            # Calculate loss

            full_loss = self.criterion(out, loss_inp)['loss']

            # Backward
            full_loss.backward()

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)

            self.optimizer.step()

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

    def get_polys(self, prediction, image_instances):

        prediction = prediction[:, 0, :, :]

        polys = []
        for idx in range(prediction.shape[0]):
            pred = prediction[idx]
            target_instance = image_instances[idx]

            mask = SegmentationMaskResult(
                prediction_labels=pred,
                coords=target_instance.coords,
                scale=target_instance.scale,
            )

            postprocessor_result = self.postprocessor.run(args=mask)
            bboxes = [np.array(prediction.coords) for prediction in postprocessor_result]
            polys.append(bboxes)

        return polys

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

        self.hmean = torch.zeros(size=(1,), dtype=torch.float32).to(self.device)
        self.precision_iou = torch.zeros(size=(1,), dtype=torch.float32).to(self.device)
        self.recall_iou = torch.zeros(size=(1,), dtype=torch.float32).to(self.device)

        with torch.no_grad():
            # Iter by loop
            for i, (batch) in enumerate(loop):

                # Forward step
                preds, loss_inp, image_instances, target_polys = self._forward_step(model, batch)

                precision, recall = self.calculate_metrics(preds[:, 0, :, :], loss_inp['shrink_map'])

                # Calculate loss
                full_loss = self.criterion(preds, loss_inp)

                # Update loss
                self.eval_loss += full_loss['loss'].detach().cpu().item()

                # Set loss value for loop
                loop.set_postfix({"loss": float(self.eval_loss / (i+1))})

                predict_polys = self.get_polys(preds, image_instances)

                for target, predict, instance in zip(target_polys, predict_polys, image_instances):
                    metric = HmeanIOUMetric(target, predict)
                    # self.save_image(hmean, instance, predict)
                    hmean.append(metric['hmean'])
                    precision.append(metric['precision'])
                    recall.append(metric['recall'])

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

        metrics = [self.f1_score.item(), self.precision.item(), self.recall.item()]
        metrics_name = ['F1', 'precision', 'recall']

        self.logger.log_epoch_metrics(
            metrics,
            metrics_name,
            self.current_epoch
        )

    def _forward_step(
            self,
            model: torch.nn.Module,
            batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor,
                         torch.Tensor, torch.Tensor, List[ImageResizerResult], List[List[np.ndarray]]]
    ) -> Tuple[torch.Tensor, Dict, List[ImageResizerResult], List[List[np.ndarray]]]:
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
        image_batch, shrink_maps_batch, shrink_masks_batch, \
        threshold_maps_batch, threshold_masks_batch, image_instances, target_polys = batch

        image_batch = image_batch.to(self.device)
        image_batch.div_(255.)
        image_batch.sub_(self.mean)
        image_batch.div_(self.std)

        shrink_maps_batch = shrink_maps_batch.to(self.device)
        shrink_masks_batch = shrink_masks_batch.to(self.device)
        threshold_maps_batch = threshold_maps_batch.to(self.device)
        threshold_masks_batch = threshold_masks_batch.to(self.device)

        out = model(image_batch)

        loss_inp = {
            'shrink_map': shrink_maps_batch,
            'shrink_mask': shrink_masks_batch,
            'threshold_map': threshold_maps_batch,
            'threshold_mask': threshold_masks_batch
        }

        return out, loss_inp, image_instances, target_polys

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

    def convert_weights(self) -> None:
        """Average the weights of the best K models and trace the model for serving.

       This method averages the weights of the best K models based on the WAR and CAR
       metrics, and then traces the model using the PyTorch JIT library. It also logs
       the traced model to MlFlow.
       """

        self.config["train"]["use_ddp"] = False
        self.model = average_weights(self.config, self.best_checkpoints)
        tracing_weights(self.model, self.config)

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
    from utils.tools import set_random_seed, get_config, save_json
    set_random_seed()

    config, task = get_config()
    trainer = Trainer(config)
    trainer.train()