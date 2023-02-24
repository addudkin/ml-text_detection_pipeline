from datasets.utils.make_border_map import MakeBorderMap
from datasets.utils.make_shrink_map import MakeShrinkMap


class TargetCreator:
    def __init__(self, mask_shrinker_cfg, border_creator_cfg):
        self.border_creator = MakeBorderMap(**border_creator_cfg)
        self.mask_shrinker = MakeShrinkMap(**mask_shrinker_cfg)

    def create_target(self, image, word_polygons):
        res = self.mask_shrinker({
            'img': image,
            'text_polys': word_polygons,
            'ignore_tags': [0] * len(word_polygons)
        })

        shrink_map = res['shrink_map']
        shrink_mask = res['shrink_mask']

        res = self.border_creator({
            'img': image,
            'text_polys': word_polygons,
            'ignore_tags': [0] * len(word_polygons)
        })

        threshold_map = res['threshold_map']
        threshold_mask = res['threshold_mask']

        return shrink_map, shrink_mask, threshold_map, threshold_mask