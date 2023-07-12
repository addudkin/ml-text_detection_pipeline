import torch

from typing import List, Union


def collate_fnc(batch) -> List[Union[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Combines a list of samples into a single batch.

    This function takes a list of samples and combines them into a single batch by padding the text samples,
    stacking the image tensors, and collecting the meta information.

    Args:
        batch: A list of samples, where each sample is a dictionary containing 'image', 'text', 'length', and 'meta' keys.

    Returns:
        A list containing the stacked image tensor, padded text tensor, list of lengths, and list of meta information.
    """

    image_batch = torch.empty(size=[len(batch), *batch[0]['image'].size()])
    shrink_maps_batch = torch.empty(size=[len(batch), *batch[0]['shrink_maps'].size()])
    shrink_masks_batch = torch.empty(size=[len(batch), *batch[0]['shrink_masks'].size()])
    threshold_maps_batch = torch.empty(size=[len(batch), *batch[0]['threshold_maps'].size()])
    threshold_masks_batch = torch.empty(size=[len(batch), *batch[0]['threshold_masks'].size()])
    image_instance_list = []
    polys_list = []

    for num, item in enumerate(batch):
        image_batch[num] = item['image']
        shrink_maps_batch[num] = item['shrink_maps']
        shrink_masks_batch[num] = item['shrink_masks']
        threshold_maps_batch[num] = item['threshold_maps']
        threshold_masks_batch[num] = item['threshold_masks']
        image_instance_list.append(item['image_instance'])
        polys_list.append(item['polys'])

    return [
        image_batch,
        shrink_maps_batch,
        shrink_masks_batch,
        threshold_maps_batch,
        threshold_masks_batch,
        image_instance_list,
        polys_list
    ]


def collate_gt(batch):
    image_batch = torch.empty(size=[len(batch), *batch[0]['image'].size()])
    image_instance_list = list()
    text_polys_list = list()

    for num, item in enumerate(batch):
        image_batch[num] = item['image']
        image_instance_list.append(item['image_instance'])
        text_polys_list.append(item['text_polys'])

    return image_batch, image_instance_list, text_polys_list