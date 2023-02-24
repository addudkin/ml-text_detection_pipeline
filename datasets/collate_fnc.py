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
    out_gt_polygons = []

    for num, item in enumerate(batch):
        image_batch[num] = item['image']
        shrink_maps_batch[num] = item['shrink_maps']
        shrink_masks_batch[num] = item['shrink_masks']
        threshold_maps_batch[num] = item['threshold_maps']
        threshold_masks_batch[num] = item['threshold_masks']
        out_gt_polygons.append(item['gt_polygons'])

    return [
        image_batch,
        shrink_maps_batch,
        shrink_masks_batch,
        threshold_maps_batch,
        threshold_masks_batch,
        out_gt_polygons
    ]


def collate_syntV2(batch) -> List[Union[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Combines a list of samples into a single batch.

    This function takes a list of samples and combines them into a single batch by padding the text samples,
    stacking the image tensors, and collecting the meta information.

    Args:
        batch: A list of samples, where each sample is a dictionary containing 'image', 'text', 'length', and 'meta' keys.

    Returns:
        A list containing the stacked image tensor, padded text tensor, list of lengths, and list of meta information.
    """
    image_batch = torch.empty(size=[len(batch), *batch[0]['image'].size()])
    for num, item in enumerate(batch):
        image_batch[num] = item['image']

    shrink_maps_batch_even = torch.empty(size=[len(batch), *batch[0]['target_even'][0].size()])
    shrink_masks_batch_even = torch.empty(size=[len(batch), *batch[0]['target_even'][1].size()])
    threshold_maps_batch_even = torch.empty(size=[len(batch), *batch[0]['target_even'][2].size()])
    threshold_masks_batch_even = torch.empty(size=[len(batch), *batch[0]['target_even'][3].size()])
    polygons_even = []

    shrink_maps_batch_odd = torch.empty(size=[len(batch), *batch[0]['target_odd'][0].size()])
    shrink_masks_batch_odd = torch.empty(size=[len(batch), *batch[0]['target_odd'][1].size()])
    threshold_maps_batch_odd = torch.empty(size=[len(batch), *batch[0]['target_odd'][2].size()])
    threshold_masks_batch_odd = torch.empty(size=[len(batch), *batch[0]['target_odd'][3].size()])
    polygons_odd = []

    for num, item in enumerate(batch):
        _, target_even, target_odd, gt_polygons = item.values()
        shrink_maps_batch_even[num] = target_even[0]
        shrink_masks_batch_even[num] = target_even[1]
        threshold_maps_batch_even[num] = target_even[2]
        threshold_masks_batch_even[num] = target_even[3]
        polygons_even.append(gt_polygons[0])

        shrink_maps_batch_odd[num] = target_odd[0]
        shrink_masks_batch_odd[num] = target_odd[1]
        threshold_maps_batch_odd[num] = target_odd[2]
        threshold_masks_batch_odd[num] = target_odd[3]
        polygons_odd.append(gt_polygons[1])


    targets = [
        [
        shrink_maps_batch_even,
        shrink_masks_batch_even,
        threshold_maps_batch_even,
        threshold_masks_batch_even,
        polygons_even
    ],
        [
        shrink_maps_batch_odd,
        shrink_masks_batch_odd,
        threshold_maps_batch_odd,
        threshold_masks_batch_odd,
        polygons_odd
        ]
    ]
    return [
        image_batch,
        targets
    ]