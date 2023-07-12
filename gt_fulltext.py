import os
import torch
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

from models import get_model
from train import Trainer
from clearml import Dataset
from typing import Tuple, List
from utils.metrics import FullTextPostProcessor
from utils.tools import get_config, get_device
from utils.map import calculate_map
from datasets import get_dataset, get_dataloader
from utils.types import SegmentationMaskResult
from utils.hmean import HmeanIOUMetric
from utils.metrics import draw_polygons
from utils.main_postprocess import SegPostprocessing


class GTEvaluater(Trainer):
    """
    Eval TD
    """
    def __init__(self, config):
        super(Trainer, self).__init__()

        self.config = config
        # Init base train element
        self.device = get_device(config)
        self.model = get_model(config, self.device).to(self.device)

        gt_dataset = get_dataset("gt", config)
        self.val_loader = get_dataloader(gt_dataset, config)

        self.postprocessor = FullTextPostProcessor(**config['post_processor']['params'])
        self.main_postprocess = SegPostprocessing(**config['post_processor']['params'])
        self.mean = torch.FloatTensor(config.data.mean).view(-1, 1, 1).to(self.device)
        self.std = torch.FloatTensor(config.data.std).view(-1, 1, 1).to(self.device)


    def get_bounding_box(self, polygon):
        x, y, w, h = cv2.boundingRect(polygon)
        top_left = [x, y]
        top_right = [x + w, y]
        bottom_right = [x + w, y + h]
        bottom_left = [x, y + h]
        return np.array([top_left, top_right, bottom_right, bottom_left])

    def _forward_step(
            self,
            model: torch.nn.Module,
            batch: Tuple[torch.Tensor, List, List]
    ) -> Tuple[torch.Tensor, List, List]:
        image_batch, image_instances, text_polys = batch

        image_batch = image_batch.to(self.device)
        image_batch.div_(255.)
        image_batch.sub_(self.mean)
        image_batch.div_(self.std)

        out = model(image_batch)

        return out, image_instances, text_polys

    def save_image(self, metric, instance, predict):
        image = instance.image
        image = image.astype(np.uint8)
        image = draw_polygons(image, predict)
        name = f"data/gt_metrics/{metric['hmean']}.png"
        image.save(name)

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

        hmean = []
        precision = []
        recall = []

        with torch.no_grad():
            # Iter by loop
            for i, (batch) in enumerate(loop):

                # Forward step
                out, image_instances, text_polys = self._forward_step(self.model, batch)

                predict_polys = self.get_polys(out, image_instances)

                for target, predict, instance in zip(text_polys, predict_polys, image_instances):
                    metric = HmeanIOUMetric(target, predict)
                    # self.save_image(hmean, instance, predict)
                    hmean.append(metric['hmean'])
                    precision.append(metric['precision'])
                    recall.append(metric['recall'])

        print(f'Средний hmean' - np.mean(hmean))
        print(f'Средний precision' - np.mean(precision))
        print(f'Средний recall' - np.mean(recall))



if __name__ == '__main__':
    config, task = get_config()
    print(config['description']['gt_dataset_id'])
    datasets = Dataset.get(dataset_id=config['description']['gt_dataset_id'])
    path2dataset = datasets.get_local_copy()
    print(f'Dataset saved into {path2dataset}')
    config['data']['datasets']['pervichka']['images_folder'] = os.path.join(path2dataset, 'images')
    config['data']['datasets']['pervichka']['annotation_folder'] = path2dataset

    annotation_name = [i for i in os.listdir(path2dataset) if i.endswith('json')]
    if len(annotation_name) > 1:
        print('Внутри много json-ов')
    else:
        annotation_name = annotation_name[0]

    config['data']['sample_name']['gt'] = annotation_name

    trainer = GTEvaluater(config)
    trainer.evaluate()