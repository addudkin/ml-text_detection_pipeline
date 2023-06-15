import torch.distributed as dist

import torch

from tqdm import tqdm
from omegaconf import DictConfig
from typing import Tuple, List, Dict
from logger import Logger

from utils.tools import get_config
from criterions import get_criterion
from train_line_segm import TrainerLD


class EvalLS(TrainerLD):
    """
    Eval LS
    """
    def evaluate(self,) -> None:
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
        self.model.eval()

        # Set eval loss counter
        self.eval_loss = torch.scalar_tensor(0).cpu()

        loop = tqdm(self.val_loader, desc='Evaluate')

        self.precision = torch.zeros(size=(1,), dtype=torch.float32).to(self.device)
        self.recall = torch.zeros(size=(1,), dtype=torch.float32).to(self.device)
        self.f1_score = torch.zeros(size=(1,), dtype=torch.float32).to(self.device)

        with torch.no_grad():
            # Iter by loop
            for i, (batch) in enumerate(loop):

                # Forward step
                out, loss_inp = self._forward_step(self.model, batch)

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
            self.eval_loss = self.eval_loss / len(self.val_loader)
            print(f"evaluate loss - {self.eval_loss}")
            print(f"PRECISION - {self.precision}")
            print(f"RECALL - {self.recall}")
            print(f"F1_SCORE - {self.f1_score}")


if __name__ == "__main__":

    config = get_config()
    trainer = EvalLS(config)
    trainer.evaluate()