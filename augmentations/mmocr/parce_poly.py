import os

import numpy as np

from PIL import Image
from shapely.geometry import Polygon, mapping

from typing import List, Dict, Union, Any
from tqdm import tqdm


def poly2shapely(polygon) -> Polygon:
    """
    Convert a polygon to shapely.geometry.Polygon.
    """
    polygon = np.array(polygon, dtype=np.float32)
    assert polygon.size % 2 == 0 and polygon.size >= 6

    polygon = polygon.reshape([-1, 2])
    return Polygon(polygon)


def get_coords(ob: dict, type_poly: str) -> List[List[float]]:
    """
    Extract polygons from ob:dict
    """
    coordinates = ob['coordinates']
    valid_polygons = []
    tuple_poly = None
    for i in coordinates:
        if type_poly == 'Polygon':
            tuple_poly = i
        elif type_poly == 'MultiPolygon':
            tuple_poly = i[0]
        else:
            assert type(tuple_poly) is None, f'{type_poly} unrecognized poly type'
        cur_poly = []
        for x, y in tuple_poly:
            cur_poly.append(x)
            cur_poly.append(y)
        valid_polygons.append(cur_poly)
    return valid_polygons


def parce(poly: Polygon) -> List[List[float]]:
    """
    Make input polygon valid
    """

    # Get structure for extracting polygons
    poly_mapped = mapping(poly)

    # Extract polygons for poly_mapped
    new_polies = []
    poly_type = poly_mapped['type']
    if poly_type in ('Polygon', 'MultiPolygon'):
        valid_polygons = get_coords(poly_mapped, poly_type)
        new_polies.extend(valid_polygons)

    elif poly_mapped['type'] == 'GeometryCollection':
        for ob in poly_mapped['geometries']:
            poly_type = ob['type']
            if poly_type in ('Polygon', 'MultiPolygon'):
                valid_polygons = get_coords(ob, poly_type)
                new_polies.extend(valid_polygons)
            else:
                pass  # MultiLineString or LineString
    else:
        pass  # MultiLineString or LineString
    return new_polies


def run(markup: dict, path2images: str) -> List[Dict[str, Union[List[Union[List[float], List[int]]], Any]]]:
    """
    Create valid markup from input data

    :return: {
        img_name: list polygons
    }

    """
    res = []
    for name, annot in tqdm(markup.items()):
        current_image = {
            'img_name': name
        }

        # Get annot and image
        bboxes = annot['boxes']
        path2image = os.path.join(path2images, name)
        image = Image.open(path2image)
        width, height = image.size

        # Check polygons
        valid_polygons = []

        # For every box (polygons) create coord and check valid
        for box in bboxes:
            poly = np.array(box)
            poly[:, 0] = poly[:, 0] * width
            poly[:, 1] = poly[:, 1] * height
            polygon = [int(i) for i in poly.flatten()]

            # Create shapely polygon for checking
            shap_poly = poly2shapely(polygon)
            if shap_poly.is_valid == False:
                # Get valid polygons for shap_poly
                polygons = make_poly_valid(shap_poly)
                valid_polygons.extend(polygons)
            else:
                valid_polygons.append(polygon)

        current_image['poly'] = valid_polygons
        res.append(current_image)
    return res


if __name__ == "__main__":
    path2sample = 'data_examples/make_poly_valid_sample/annotation.json'
    path2images = 'data_examples/make_poly_valid_sample/test_sample/images'
    test_sample = load_json(path2sample)

    result = run(test_sample, path2images)
