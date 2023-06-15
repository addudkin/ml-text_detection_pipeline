import random
import numpy as np

from typing import Tuple, Sequence, Dict

from numpy import ndarray

from augmentations.mmocr import crop_polygon, poly2bbox


class TextDetRandomCrop:
    """Randomly select a region and crop images to a target size and make sure
    to contain text region. This transform may break up text instances, and for
    broken text instances, we will crop it's bbox and polygon coordinates. This
    transform is recommend to be used in segmentation-based network.

    Required Keys:

    - img
    - gt_polygons
    - gt_bboxes
    - gt_bboxes_labels
    - gt_ignored

    Modified Keys:

    - img
    - img_shape
    - gt_polygons
    - gt_bboxes
    - gt_bboxes_labels
    - gt_ignored

    Args:
        target_size (tuple(int, int) or int): Target size for the cropped
            image. If it's a tuple, then target width and target height will be
            ``target_size[0]`` and ``target_size[1]``, respectively. If it's an
            integer, them both target width and target height will be
            ``target_size``.
        positive_sample_ratio (float): The probability of sampling regions
            that go through text regions. Defaults to 5. / 8.
    """

    def __init__(self,
                 target_size: Tuple[int, int] or int,
                 positive_sample_ratio: float = 5.0 / 8.0) -> None:
        self.target_size = target_size if isinstance(
            target_size, tuple) else (target_size, target_size)
        self.positive_sample_ratio = positive_sample_ratio

    def _get_postive_prob(self) -> float:
        """Get the probability to do positive sample.

        Returns:
            float: The probability to do positive sample.
        """
        return np.random.random_sample()

    def _sample_num(self, start, end):
        """Sample a number in range [start, end].

        Args:
            start (int): Starting point.
            end (int): Ending point.

        Returns:
            (int): Sampled number.
        """
        return random.randint(start, end)

    def _sample_offset(self, gt_polygons: Sequence[np.ndarray],
                       img_size: Tuple[int, int]) -> Tuple[int, int]:
        """Samples the top-left coordinate of a crop region, ensuring that the
        cropped region contains at least one polygon.

        Args:
            gt_polygons (list(ndarray)) : Polygons.
            img_size (tuple(int, int)) : Image size in the format of
                (height, width).

        Returns:
            tuple(int, int): Top-left coordinate of the cropped region.
        """
        h, w = img_size
        t_w, t_h = self.target_size

        # target size is bigger than origin size
        t_h = t_h if t_h < h else h
        t_w = t_w if t_w < w else w
        if (gt_polygons is not None and len(gt_polygons) > 0
                and self._get_postive_prob() < self.positive_sample_ratio):

            # make sure to crop the positive region

            # the minimum top left to crop positive region (h,w)
            tl = np.array([h + 1, w + 1], dtype=np.int32)
            for gt_polygon in gt_polygons:
                temp_point = np.min(gt_polygon.reshape(2, -1), axis=1)
                if temp_point[0] <= tl[0]:
                    tl[0] = temp_point[0]
                if temp_point[1] <= tl[1]:
                    tl[1] = temp_point[1]
            tl = tl - (t_h, t_w)
            tl[tl < 0] = 0
            # the maximum bottum right to crop positive region
            br = np.array([0, 0], dtype=np.int32)
            for gt_polygon in gt_polygons:
                temp_point = np.max(gt_polygon.reshape(2, -1), axis=1)
                if temp_point[0] > br[0]:
                    br[0] = temp_point[0]
                if temp_point[1] > br[1]:
                    br[1] = temp_point[1]
            br = br - (t_h, t_w)
            br[br < 0] = 0

            # if br is too big so that crop the outside region of img
            br[0] = min(br[0], h - t_h)
            br[1] = min(br[1], w - t_w)
            #
            h = self._sample_num(tl[0], br[0]) if tl[0] < br[0] else 0
            w = self._sample_num(tl[1], br[1]) if tl[1] < br[1] else 0
        else:
            # make sure not to crop outside of img

            h = self._sample_num(0, h - t_h) if h - t_h > 0 else 0
            w = self._sample_num(0, w - t_w) if w - t_w > 0 else 0

        return (h, w)

    def _crop_img(self, img: np.ndarray, offset: Tuple[int, int],
                  target_size: Tuple[int, int]):
        """Crop the image given an offset and a target size.

        Args:
            img (ndarray): Image.
            offset (Tuple[int. int]): Coordinates of the starting point.
            target_size: Target image size.
        """
        h, w = img.shape[:2]
        target_size = target_size[::-1]
        br = np.min(
            np.stack((np.array(offset) + np.array(target_size), np.array(
                (h, w)))),
            axis=0)
        return img[offset[0]:br[0], offset[1]:br[1]], np.array(
            [offset[1], offset[0], br[1], br[0]])

    def _crop_polygons(self, polygons: Sequence[np.ndarray],
                       crop_bbox: np.ndarray) -> Sequence[np.ndarray]:
        """Crop polygons to be within a crop region. If polygon crosses the
        crop_bbox, we will keep the part left in crop_bbox by cropping its
        boardline.

        Args:
            polygons (list(ndarray)): List of polygons [(N1, ), (N2, ), ...].
            crop_bbox (ndarray): Cropping region. [x1, y1, x2, y1].

        Returns
            tuple(List(ArrayLike), list[int]):
                - (List(ArrayLike)): The rest of the polygons located in the
                    crop region.
                - (list[int]): Index list of the reserved polygons.
        """
        polygons_cropped = []
        kept_idx = []
        for idx, polygon in enumerate(polygons):
            if polygon.size < 6:
                continue
            polys = crop_polygon(polygon, crop_bbox)
            if polys is not None:
                for poly in polys:
                    poly = poly.reshape(-1, 2) - (crop_bbox[0], crop_bbox[1])
                    polygons_cropped.append(poly.reshape(-1))
                    kept_idx.append(idx)
        return (polygons_cropped, kept_idx)

    def transform(self, results: Dict) -> Dict:
        """Applying random crop on results.
        Args:
            results (dict): Result dict contains the data to transform.

        Returns:
            dict: The transformed data
        """
        if self.target_size == results['img'].shape[:2][::-1]:
            return results
        gt_polygons = results['gt_polygons']
        crop_offset = self._sample_offset(gt_polygons,
                                          results['img'].shape[:2])

        img, crop_bbox = self._crop_img(results['img'], crop_offset,
                                        self.target_size)
        results['img'] = img
        results['img_shape'] = img.shape[:2]
        gt_polygons, polygon_kept_idx = self._crop_polygons(
            gt_polygons, crop_bbox)
        bboxes = [poly2bbox(poly) for poly in gt_polygons]
        results['gt_bboxes'] = np.array(
            bboxes, dtype=np.float32).reshape(-1, 4)

        results['gt_polygons'] = gt_polygons
        results['gt_bboxes_labels'] = results['gt_bboxes_labels'][
            polygon_kept_idx]
        results['gt_ignored'] = results['gt_ignored'][polygon_kept_idx]
        return results


class TextDetRandomCropV2:
    """Randomly select a region and crop images to a target size and make sure
    to contain text region. This transform may break up text instances, and for
    broken text instances, we will crop it's bbox and polygon coordinates. This
    transform is recommend to be used in segmentation-based network.

    Required Keys:

    - img
    - gt_polygons
    - gt_bboxes
    - gt_bboxes_labels
    - gt_ignored

    Modified Keys:

    - img
    - img_shape
    - gt_polygons
    - gt_bboxes
    - gt_bboxes_labels
    - gt_ignored

    Args:
        target_size (tuple(int, int) or int): Target size for the cropped
            image. If it's a tuple, then target width and target height will be
            ``target_size[0]`` and ``target_size[1]``, respectively. If it's an
            integer, them both target width and target height will be
            ``target_size``.
        positive_sample_ratio (float): The probability of sampling regions
            that go through text regions. Defaults to 5. / 8.
    """

    def __init__(self,
                 target_size: Tuple[int, int] or int,
                 positive_sample_ratio: float = 5.0 / 8.0) -> None:
        self.target_size = target_size if isinstance(
            target_size, tuple) else (target_size, target_size)
        self.positive_sample_ratio = positive_sample_ratio

    def _get_postive_prob(self) -> float:
        """Get the probability to do positive sample.

        Returns:
            float: The probability to do positive sample.
        """
        return np.random.random_sample()

    def _sample_num(self, start, end):
        """Sample a number in range [start, end].

        Args:
            start (int): Starting point.
            end (int): Ending point.

        Returns:
            (int): Sampled number.
        """
        return random.randint(start, end)

    def _sample_offset(self, gt_polygons: Sequence[np.ndarray],
                       img_size: Tuple[int, int]) -> Tuple[int, int]:
        """Samples the top-left coordinate of a crop region, ensuring that the
        cropped region contains at least one polygon.

        Args:
            gt_polygons (list(ndarray)) : Polygons.
            img_size (tuple(int, int)) : Image size in the format of
                (height, width).

        Returns:
            tuple(int, int): Top-left coordinate of the cropped region.
        """
        h, w = img_size
        t_w, t_h = self.target_size

        # target size is bigger than origin size
        t_h = t_h if t_h < h else h
        t_w = t_w if t_w < w else w
        if (gt_polygons is not None and len(gt_polygons) > 0
                and self._get_postive_prob() < self.positive_sample_ratio):

            # make sure to crop the positive region

            # the minimum top left to crop positive region (h,w)
            tl = np.array([h + 1, w + 1], dtype=np.int32)
            for gt_polygon in gt_polygons:
                temp_point = np.min(gt_polygon.reshape(2, -1), axis=1)
                if temp_point[0] <= tl[0]:
                    tl[0] = temp_point[0]
                if temp_point[1] <= tl[1]:
                    tl[1] = temp_point[1]
            tl = tl - (t_h, t_w)
            tl[tl < 0] = 0
            # the maximum bottum right to crop positive region
            br = np.array([0, 0], dtype=np.int32)
            for gt_polygon in gt_polygons:
                temp_point = np.max(gt_polygon.reshape(2, -1), axis=1)
                if temp_point[0] > br[0]:
                    br[0] = temp_point[0]
                if temp_point[1] > br[1]:
                    br[1] = temp_point[1]
            br = br - (t_h, t_w)
            br[br < 0] = 0

            # if br is too big so that crop the outside region of img
            br[0] = min(br[0], h - t_h)
            br[1] = min(br[1], w - t_w)
            #
            h = self._sample_num(tl[0], br[0]) if tl[0] < br[0] else 0
            w = self._sample_num(tl[1], br[1]) if tl[1] < br[1] else 0
        else:
            # make sure not to crop outside of img

            h = self._sample_num(0, h - t_h) if h - t_h > 0 else 0
            w = self._sample_num(0, w - t_w) if w - t_w > 0 else 0

        return (h, w)

    def _crop_img(self, img: np.ndarray, offset: Tuple[int, int],
                  target_size: Tuple[int, int]):

        """Crop the image given an offset and a target size.

        Args:
            img (ndarray): Image.
            offset (Tuple[int. int]): Coordinates of the starting point.
            target_size: Target image size.
        """
        h, w = img.shape[:2]
        target_size = target_size[::-1]
        br = np.min(
            np.stack((np.array(offset) + np.array(target_size), np.array(
                (h, w)))),
            axis=0)
        return img[offset[0]:br[0], offset[1]:br[1]], np.array(
            [offset[1], offset[0], br[1], br[0]])

    def _crop_polygons(self, polygons: Sequence[np.ndarray],
                       crop_bbox: np.ndarray) -> Sequence[np.ndarray]:
        """Crop polygons to be within a crop region. If polygon crosses the
        crop_bbox, we will keep the part left in crop_bbox by cropping its
        boardline.

        Args:
            polygons (list(ndarray)): List of polygons [(N1, ), (N2, ), ...].
            crop_bbox (ndarray): Cropping region. [x1, y1, x2, y1].

        Returns
            tuple(List(ArrayLike), list[int]):
                - (List(ArrayLike)): The rest of the polygons located in the
                    crop region.
                - (list[int]): Index list of the reserved polygons.
        """
        polygons_cropped = []
        kept_idx = []
        for idx, polygon in enumerate(polygons):
            if polygon.size < 6:
                continue
            polys = crop_polygon(polygon, crop_bbox)
            if polys is not None:
                for poly in polys:
                    poly = poly.reshape(-1, 2) - (crop_bbox[0], crop_bbox[1])
                    polygons_cropped.append(poly.reshape(-1))
                    kept_idx.append(idx)
        return (polygons_cropped, kept_idx)

    def transform(self, results: Dict) -> Dict:
        """Applying random crop on results.
        Args:
            results (dict): Result dict contains the data to transform.

        Returns:
            dict: The transformed data
        """
        if self.target_size == results['img'].shape[:2][::-1]:
            return results
        list_crop_offset = []
        for gt_polygons in results['gt_polygons']:
            list_crop_offset.extend(gt_polygons)

        crop_offset = self._sample_offset(list_crop_offset,
                                          results['img'].shape[:2])

        img, crop_bbox = self._crop_img(results['img'], crop_offset,
                                        self.target_size)
        results['img'] = img
        results['img_shape'] = img.shape[:2]

        result_polys = []
        for gt_polygons in results['gt_polygons']:
            gt_polygons, polygon_kept_idx = self._crop_polygons(
                gt_polygons, crop_bbox)

            result_polys.append(gt_polygons)
        results['gt_polygons'] = result_polys
        return results



if __name__ == '__main__':
    import pickle

    path2data = 'data_test'

    file = open(f"{path2data}/image.pkl", 'rb')
    image = pickle.load(file)

    file = open(f"{path2data}/polys.pkl", 'rb')
    polys = pickle.load(file)
    print()