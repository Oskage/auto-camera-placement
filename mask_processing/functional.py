import os
import os.path as osp
from collections import namedtuple
from math import ceil
from enum import IntEnum

from skimage import draw
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib


class Label(IntEnum):
    wall = 1
    door = 2
    window = 3


Point = namedtuple('Point', 'x y')


class PossibleLocations:
    MIN_RATING = 0
    MAX_RATING = 50
    DEFAULT_RATING = 25

    def __init__(self, points: tuple[np.ndarray, np.ndarray]):
        self.points = points
        self.rating = {(y, x): PossibleLocations.DEFAULT_RATING for (y, x) in zip(*points)}

    def __str__(self):
        parts = list()
        parts.append('===== Point =====\n')
        for point in self.rating:
            parts.append(f'(x={point[1]}, y={point[0]}) -> {self.rating[(point[0], point[1])]}\n')
        parts.append('=================\n')
        return ''.join(parts)

    def adjust_rating(self, where: tuple[np.ndarray, np.ndarray], rating_delta: int):
        for (y, x) in zip(*where):
            try:
                rating = self.rating[(y, x)] + rating_delta
                rating = max(PossibleLocations.MIN_RATING,
                             min(PossibleLocations.MAX_RATING, rating))
                self.rating[(y, x)] = rating
            except KeyError:
                pass


def load_masks(folder_path: str) -> list[np.ndarray]:
    masks = []
    files = [file for file in os.listdir(folder_path) if file.endswith('.png')]
    for filename in files:
        filepath = osp.join(folder_path, filename)
        mask = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
        assert mask is not None, f'File {filepath} was not loaded'
        masks.append(mask)
    return masks


def plot_gray_images(image: np.ndarray, *images: np.ndarray):
    if images:
        images = [image] + list(images)
    else:
        images = [image]

    ncols = 2
    nrows = ceil(len(images) / 2)
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 10, nrows * 10))
    axes = np.ravel(axes)

    for idx, image in enumerate(images):
        assert len(image.shape) == 2, (f'Got{image.shape=}, but image '
                                       f'must be two dimensional')
        axes[idx].imshow(image, cmap='gray')

    if len(axes) != len(images):
        axes[-1].imshow(np.zeros_like(images[-1]), cmap='gray')

    plt.show()


def plot_location(primitive_mask: np.ndarray, locations: PossibleLocations):
    cmap = matplotlib.cm.get_cmap('RdBu')

    mask = primitive_mask / 4
    image = np.stack((mask, mask, mask), axis=2)

    for (y, x) in zip(*locations.points):
        color = locations.rating[(y, x)] / PossibleLocations.MAX_RATING
        image[y, x] = cmap(color)[:-1]

    fig, axes = plt.subplots(figsize=(15, 15))
    axes.imshow(image)
    plt.show()


def get_rectangle_boundary(
        start: tuple[int, int],
        end: tuple[int, int]
) -> tuple[np.ndarray, np.ndarray]:
    return draw.rectangle_perimeter((start[0] + 1, start[1] + 1),
                                    (end[0] - 1, end[1] - 1))


def get_label_mask(mask: np.ndarray, label: Label) -> np.ndarray:
    walls_mask = np.zeros_like(mask)
    walls_mask[mask == label.value] = 255
    return walls_mask


def get_primitive_rectangles(
        binary_mask: np.ndarray,
        rect_w: int = 5,
        rect_h: int = 5,
        frac: float = 0.1
) -> list[tuple[np.ndarray, np.ndarray]]:
    assert 0 < frac <= 1, f'Got {frac=}, but it must be in (0, 1]'

    rectangles = []
    for y in range(0, binary_mask.shape[0] - rect_h, rect_h):
        for x in range(0, binary_mask.shape[1] - rect_w, rect_w):
            object_pixels = np.sum(binary_mask[y: y + rect_h, x: x + rect_w] != 0)

            if object_pixels / (rect_w * rect_h) > frac:
                rectangles.append(get_rectangle_boundary((y, x),
                                                         (y + rect_h, x + rect_w)))

    return rectangles


def draw_primitive_rectangles(
        image: np.ndarray,
        rectangles: list[tuple[np.ndarray, np.ndarray]],
        color: int
):
    assert 0 <= color <= 255, f'Got {color=}, but it must be in [0, 255]'

    for rectangle_boundary in rectangles:
        image[rectangle_boundary] = color


# def get_primitive_mask(
#         mask: np.ndarray,
#         rect_w: int = 5,
#         rect_h: int = 5,
#         frac: float = 0.1
# ) -> np.ndarray:
#     assert 0 < frac <= 1, f'Got {frac=}, but it must be in (0, 1]'
#
#     primitive_mask = np.zeros((mask.shape[0] // rect_h, mask.shape[1] // rect_w))
#
#     for y in range(0, mask.shape[0] - rect_h, rect_h):
#         for x in range(0, mask.shape[1] - rect_w, rect_w):
#             label_counter = Counter(np.ravel(mask[y: y + rect_h, x: x + rect_w]))
#
#             # Remove background
#             if label_counter[0] / (rect_h * rect_w) > 1 - frac:
#                 continue
#
#             most_common_label = label_counter.most_common(1)[0][0]
#             primitive_mask[y // rect_h, x // rect_w] = most_common_label
#
#     return primitive_mask


def get_primitive_mask_by_rectangles(
        mask: np.ndarray,
        rectangles: dict[Label, list[tuple[np.ndarray, np.ndarray]]],
        rect_w: int = 5,
        rect_h: int = 5
) -> np.ndarray:
    primitive_mask = np.zeros((mask.shape[0] // rect_h, mask.shape[1] // rect_w))
    for label in Label:
        for rectangle in rectangles[label]:
            primitive_mask[rectangle[0][0] // rect_h, rectangle[1][0] // rect_w] = label.value
    return primitive_mask


def get_4_neighbors(mask: np.ndarray, x: int, y: int) -> set[Point]:
    min_height, min_width = 0, 0
    max_height, max_width = mask.shape

    neighbors = set()
    neighbors_candidates = [
        Point(x=x, y=y - 1),
        Point(x=x + 1, y=y),
        Point(x=x, y=y + 1),
        Point(x=x - 1, y=y)
    ]

    for point in neighbors_candidates:
        if (min_width <= point.x <= max_width - 1 and min_height <= point.y <= max_height - 1
                and mask[point.y, point.x] == 0):
            neighbors.add(point)

    return neighbors


def get_neighbor_pixels_of_label(
        primitive_mask: np.ndarray,
        label: Label
) -> tuple[np.ndarray, np.ndarray]:
    neighbors = set()
    for (y, x) in zip(*(primitive_mask == label).nonzero()):
        neighbors.update(get_4_neighbors(primitive_mask, x=x, y=y))

    ys = list()
    xs = list()
    for point in neighbors:
        ys.append(point.y)
        xs.append(point.x)

    return np.array(ys), np.array(xs)
