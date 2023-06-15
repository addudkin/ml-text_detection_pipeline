import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d


def roll_top_left_first(coords: np.array) -> np.array:
    for _ in range(4):
        distances = np.linalg.norm(coords, 2, axis=1)
        is_first = np.argmin(distances) == 0

        if is_first:
            break
        coords = np.roll(coords, 1, 0)
    else:
        raise Exception("Failed to find correct sort")
    return coords


def calculate_euclidean(pnt_1: np.ndarray,
                        pnt_2: np.ndarray) -> np.ndarray:
    """

    :param pnt_1:
    :param pnt_2:

    :return:
    """
    legs = np.power(pnt_1[np.newaxis] - pnt_2, 2)
    distance = np.sqrt(legs[:, 0] + legs[:, 1])

    return distance


def order_four_points(pts: np.ndarray,
                      sort_using_euclidean: bool = True) -> np.ndarray:
    """

    :param pts:
    :param sort_using_euclidean:
    :return:
    """
    # sort the points based on their x-coordinates
    x_sorted = pts[np.argsort(pts[:, 0]), :]
    # grab the left-most and right-most points from the sorted
    # x-coordinate points
    left_most = x_sorted[:2, :]
    right_most = x_sorted[2:, :]
    # now, sort the left-most coordinates according to their
    # y-coordinates so we can grab the top-left and bottom-left
    # points, respectively
    left_most = left_most[np.argsort(left_most[:, 1]), :]
    (tl, bl) = left_most
    # now that we have the top-left coordinate, use it as an
    # anchor to calculate the Euclidean distance between the
    # top-left and right-most points; by the Pythagorean
    # theorem, the point with the largest distance will be
    # our bottom-right point
    if sort_using_euclidean:
        d = calculate_euclidean(tl, right_most)
    else:
        d = right_most[:, 1]
    tr, br = right_most[np.argsort(d), :]
    # return the coordinates in top-left, top-right,
    # bottom-right, and bottom-left order
    return np.array([tl, tr, br, bl], dtype=np.float32)


def corted_coords(transformed_poly_group):
    """Сортируем строки после кропа"""
    if len(transformed_poly_group[0]) == 0 and len(transformed_poly_group[1]) != 0:
        return [transformed_poly_group[1], transformed_poly_group[0]]

    if len(transformed_poly_group[0]) != 0 and len(transformed_poly_group[1]) == 0:
        return [transformed_poly_group[0], transformed_poly_group[1]]

    if len(transformed_poly_group[0]) == 0 and len(transformed_poly_group[1]) == 0:
        return transformed_poly_group

    first_even = transformed_poly_group[0][0]
    first_odd = transformed_poly_group[1][0]

    y_even = first_even[:, 1]
    y_even = y_even.min()

    y_odd = first_odd[:, 1]
    y_odd = y_odd.min()

    if y_even < y_odd:
        even_coords = transformed_poly_group[0]
        odd_cords = transformed_poly_group[1]
    else:
        even_coords = transformed_poly_group[1]
        odd_cords = transformed_poly_group[0]
    return [even_coords, odd_cords]

def get_dots(point, top):
    rect = cv2.minAreaRect(point)
    (x, y), (h, w), angle = rect
#     if angle >= 45:
#         h, w = w, h
    box = cv2.boxPoints(((x, y), (h, w), angle))
    #box = sorted(box , key=lambda k: [k[1], k[0]])
    box = order_four_points(box)
    #box = roll_top_left_first(box)
    if top:
        return box[0], box[1]
    else:
        return box[-1], box[-2]

def get_straight_line(line):

    x_last = line[-1][0]
    y_center = (line[:, 1].max() + line[:, 1].min()) // 2
    start_dot = [0, y_center]
    end_dot = [x_last, y_center]

    new_line = np.array([start_dot, end_dot])
    return new_line

def get_straight_rest(rest, width, mode):

    if mode == 'max':
        y = rest[:, 1].max()
    elif mode == 'min':
        y = rest[:, 1].min()
    else:
        print('error')

    x0 = rest[0][0]

    start_dot = [x0, y]
    end_dot = [width, y]
    new_line = np.array([start_dot, end_dot])
    return new_line

def compare_two_rows(zero_row, first_row, width, highest):

    zero_row = zero_row[zero_row[:, 0].argsort()]
    first_row = first_row[first_row[:, 0].argsort()]

    if zero_row[-1][0] > first_row[-1][0]:
        if highest == 'even':
            mode = 'max'
        else:
            mode = 'min'
        last_x = first_row[-1][0]
        last_y = first_row[-1][1]
        rest = zero_row[zero_row[:, 0] > last_x]
        zero_row = zero_row[zero_row[:, 0] <= last_x]
    else:
        if highest == 'odd':
            mode = 'max'
        else:
            mode = 'min'
        last_x = zero_row[-1][0]
        last_y = zero_row[-1][1]
        rest = first_row[first_row[:, 0] > last_x]
        first_row = first_row[first_row[:, 0] <= last_x]

    new_line = np.vstack([zero_row, first_row])
    new_line = new_line[new_line[:, 0].argsort()]

    if rest.shape[0] > 4:
        rest = get_straight_rest(rest, width, mode)
    else:
        rest = np.array([])

    new_line = get_straight_line(new_line)

    if rest.shape[0] > 0:
        new_line = np.vstack([new_line, rest])
        first_dot = new_line[0]
        last_dot = new_line[-1]

        new_line[:, 0] = gaussian_filter1d(new_line[:, 0], 1.0)
        new_line[:, 1] = gaussian_filter1d(new_line[:, 1], 1.0)
        new_line = np.vstack([np.array([0, first_dot[1]]),
                              new_line,
                              np.array([width, last_dot[1]])]
                             )
    else:
        first_dot = new_line[0]
        last_dot = new_line[-1]
        new_line = np.vstack([np.array([0, first_dot[1]]),
                              new_line,
                              np.array([width, last_dot[1]])]
                             )
    return new_line


def get_row_line_first(line_even, line_odd, width, sigma=1.5):

    even_row = np.vstack([get_dots(i, top=False) for i in line_even])
    odd_row = np.vstack([get_dots(i, top=True) for i in line_odd])

    ## even сверху
    new_line = compare_two_rows(even_row, odd_row, width, 'even')

    return new_line


def get_row_line_second(line_even, line_odd, width, sigma=1.5):
    even_row = np.vstack([get_dots(i, top=True) for i in line_even])
    odd_row = np.vstack([get_dots(i, top=False) for i in line_odd])

    ## odd сверху
    new_line = compare_two_rows(even_row, odd_row, width, 'odd')

    return new_line


def get_smoth_signle_bottom_line(line, width):
    line = np.vstack([get_dots(i, top=False) for i in line])

    max_y = line[:, 1].max()

    new_line = np.array([[0, max_y], [width, max_y]])
    return new_line


def create_lines_mask(polys_even, polys_odd, image):
    width, height = image.shape[:2]

    mask1 = np.zeros((width, height))
    if len(polys_even) < len(polys_odd):
        print('Пум пум')

    full_lines = []
    if len(polys_even) != len(polys_odd):
        for even_line, odd_line in zip(polys_even[:-1], polys_odd):
            line = get_row_line_first(even_line, odd_line, width)
            full_lines.append(line)

        for even_line, odd_line in zip(polys_even[1:], polys_odd):
            line = get_row_line_second(even_line, odd_line, width)
            full_lines.append(line)

        full_lines.append(get_smoth_signle_bottom_line(polys_even[-1], width))
    else:
        for even_line, odd_line in zip(polys_even, polys_odd):
            line = get_row_line_first(even_line, odd_line, width)
            full_lines.append(line)

        for even_line, odd_line in zip(polys_even[1:], polys_odd[:-1]):
            line = get_row_line_second(even_line, odd_line, width)
            full_lines.append(line)

        full_lines.append(get_smoth_signle_bottom_line(polys_odd[-1], width))

    for line in full_lines:
        if line.shape[0] == 0:
            continue
        mask1 = cv2.polylines(mask1.astype(np.int32), [line.astype(np.int32)], False, color=(1, 1, 1),
                              thickness=3)

    mask2 = (mask1 == 0).astype(np.int32)

    return mask1, mask2