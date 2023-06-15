import torch

from typing import List, Union
from torch.nn.utils.rnn import pad_sequence


def collate_synt(batch) -> List[Union[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
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
    # out_gt_polygons = []

    for num, item in enumerate(batch):
        image_batch[num] = item['image']
        shrink_maps_batch[num] = item['shrink_maps']
        shrink_masks_batch[num] = item['shrink_masks']
        threshold_maps_batch[num] = item['threshold_maps']
        threshold_masks_batch[num] = item['threshold_masks']
        # out_gt_polygons.append(item['gt_polygons'])

    return [
        image_batch,
        shrink_maps_batch,
        shrink_masks_batch,
        threshold_maps_batch,
        threshold_masks_batch,
        # out_gt_polygons
    ]


def collate_text_and_lines(batch) -> List[Union[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
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
    masks_batch = torch.empty(size=[len(batch), *batch[0]['masks'].size()])

    for num, item in enumerate(batch):
        image_batch[num] = item['image']
        shrink_maps_batch[num] = item['shrink_maps']
        shrink_masks_batch[num] = item['shrink_masks']
        threshold_maps_batch[num] = item['threshold_maps']
        threshold_masks_batch[num] = item['threshold_masks']
        masks_batch[num] = item['masks']

    return [
        image_batch,
        shrink_maps_batch,
        shrink_masks_batch,
        threshold_maps_batch,
        threshold_masks_batch,
        masks_batch,
    ]


def collate_even_odd(batch) -> List[Union[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Collate for even and odd lines"""
    image_batch = torch.empty(size=[len(batch), *batch[0]['image'].size()])
    even_targets = torch.empty(size=[len(batch), *batch[0]['even_targets'].size()])
    odd_targets = torch.empty(size=[len(batch), *batch[0]['odd_targets'].size()])

    for num, item in enumerate(batch):
        image_batch[num] = item['image']
        even_targets[num] = item['even_targets']
        odd_targets[num] = item['odd_targets']

    return [
        image_batch,
        even_targets,
        odd_targets
    ]


def collate_synt_line_det(batch) -> List[Union[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Combines a list of samples into a single batch.

    This function takes a list of samples and combines them into a single batch by padding the text samples,
    stacking the image tensors, and collecting the meta information.

    Args:
        batch: A list of samples, where each sample is a dictionary containing 'image', 'text', 'length', and 'meta' keys.

    Returns:
        A list containing the stacked image tensor, padded text tensor, list of lengths, and list of meta information.
    """

    image_batch = torch.empty(size=[len(batch), *batch[0]['image'].size()])
    masks = torch.empty(size=[len(batch), *batch[0]['masks'].size()])

    for num, item in enumerate(batch):
        image_batch[num] = item['image']
        masks[num] = item['masks']

    return [
        image_batch,
        masks
    ]