import os
import os.path as osp
from collections import namedtuple
from math import ceil
from enum import IntEnum
from random import randrange

from skimage import draw
from tqdm import tqdm
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib


class Label(IntEnum):
    wall = 1
    door = 2
    window = 3


Point = namedtuple('Point', 'x y')

VALUE_PER_POINT = 25
LOCATION_COEFFICIENT = 50

CAMERA_COLOR = np.array([0, 255, 0]) / 255
VIEWED_AREA_COLOR = np.array([150, 143, 255]) / 255
DEFENSE_POINT_COLOR = np.array([255, 0, 0]) / 255


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


def plot_camera_view_area(
        primitive_mask: np.ndarray,
        view_point: Point
):
    mask = primitive_mask / 4
    image = np.stack((mask, mask, mask), axis=2)

    *viewed_points, _ = get_viewed_points_from_point(primitive_mask, view_point)

    for y, x in zip(*viewed_points):
        image[y, x] = VIEWED_AREA_COLOR

    image[view_point.y, view_point.x] = CAMERA_COLOR
    fig, axes = plt.subplots(figsize=(15, 15))
    axes.imshow(image)
    plt.show()


def plot_camera_view_with_defense_point(
        primitive_mask: np.ndarray,
        camera_location: Point,
        defense_point: Point
):
    mask = primitive_mask / 4
    image = np.stack((mask, mask, mask), axis=2)

    *viewed_points, _ = get_viewed_points_from_point(primitive_mask, camera_location)

    for y, x in zip(*viewed_points):
        image[y, x] = VIEWED_AREA_COLOR

    image[camera_location.y, camera_location.x] = CAMERA_COLOR
    image[defense_point.y, defense_point.x] = DEFENSE_POINT_COLOR
    fig, axes = plt.subplots(figsize=(15, 15))
    axes.imshow(image)
    plt.show()


def save_camera_view_with_defense_point(
        primitive_mask: np.ndarray,
        camera_location: Point,
        defense_point: Point,
        filename: str
):
    mask = primitive_mask / 4
    image = np.stack((mask, mask, mask), axis=2)

    *viewed_points, _ = get_viewed_points_from_point(primitive_mask, camera_location)

    for y, x in zip(*viewed_points):
        image[y, x] = VIEWED_AREA_COLOR

    image[camera_location.y, camera_location.x] = CAMERA_COLOR
    image[defense_point.y, defense_point.x] = DEFENSE_POINT_COLOR

    pretty_image = (image * 255).astype(np.uint8)
    pretty_image = cv2.cvtColor(pretty_image, cv2.COLOR_RGB2BGR)

    scale_percent = 500
    width = int(pretty_image.shape[1] * scale_percent / 100)
    height = int(pretty_image.shape[0] * scale_percent / 100)
    dim = (width, height)
    pretty_image = cv2.resize(pretty_image, dim, interpolation=cv2.INTER_AREA)
    cv2.imwrite(filename, pretty_image)


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


def get_primitive_mask_by_rectangles(
        mask: np.ndarray,
        rectangles: dict[Label, list[tuple[np.ndarray, np.ndarray]]],
        rect_w: int = 5,
        rect_h: int = 5
) -> np.ndarray:
    primitive_mask = np.zeros((mask.shape[0] // rect_h, mask.shape[1] // rect_w), dtype=np.uint8)
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


def remove_outer_locations(locations: PossibleLocations, primitive_mask: np.ndarray):
    contours, _ = cv2.findContours(primitive_mask, mode=cv2.RETR_EXTERNAL,
                                   method=cv2.CHAIN_APPROX_SIMPLE)

    inner_ys = []
    inner_xs = []
    for y, x in zip(*locations.points):
        for contour in contours:
            if cv2.pointPolygonTest(contour, (int(x), int(y)), measureDist=False) > 0:
                inner_ys.append(y)
                inner_xs.append(x)

    locations.points = (inner_ys, inner_xs)


def edge_pixels_of_image(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # upper -> right -> bottom -> left edge
    ys = np.concatenate((
        np.full(mask.shape[1], fill_value=0, dtype=int),
        np.arange(0, mask.shape[0], dtype=int),
        np.full(mask.shape[1], fill_value=mask.shape[0] - 1, dtype=int),
        np.arange(0, mask.shape[0], dtype=int)
    ))
    xs = np.concatenate((
        np.arange(0, mask.shape[1], dtype=int),
        np.full(mask.shape[0], fill_value=mask.shape[1] - 1, dtype=int),
        np.arange(0, mask.shape[1], dtype=int),
        np.full(mask.shape[0], fill_value=0, dtype=int)
    ))

    return ys, xs


def get_viewed_points_from_point(
        primitive_mask: np.ndarray,
        point: Point
) -> tuple[np.ndarray, np.ndarray, set[tuple[int, int]]]:
    seen = set()
    viewed_area_ys = []
    viewed_area_xs = []

    for edge_y, edge_x in zip(*edge_pixels_of_image(primitive_mask)):
        line_ys, line_xs = draw.line(point.y, point.x, edge_y, edge_x)

        for y, x in zip(line_ys, line_xs):
            if primitive_mask[y, x] != 0:
                break

            if (y, x) in seen:
                continue
            seen.add((y, x))
            viewed_area_ys.append(y)
            viewed_area_xs.append(x)

    return np.array(viewed_area_ys), np.array(viewed_area_xs), seen


def compute_location_value(
        primitive_mask: np.ndarray,
        location: Point,
        location_rating: int,
        defense_point: Point
) -> int | float:
    value = LOCATION_COEFFICIENT * location_rating

    *viewed_points, seen = get_viewed_points_from_point(primitive_mask, location)
    if (defense_point.y, defense_point.x) not in seen:
        return float('-inf')

    value += VALUE_PER_POINT * len(viewed_points[0])

    return value


def find_best_location(
        primitive_mask: np.ndarray,
        locations: PossibleLocations,
        defense_point: Point
) -> Point:
    max_value = float('-inf')
    best_location = None

    for location_y, location_x in tqdm(zip(*locations.points),
                                       total=len(locations.points[0])):
        location = Point(location_x, location_y)
        value = compute_location_value(
            primitive_mask,
            location=location,
            location_rating=locations.rating[(location_y, location_x)],
            defense_point=defense_point)

        if value > max_value:
            max_value = value
            best_location = location

    return best_location


def generate_random_defense_point(
        primitive_mask: np.ndarray
) -> Point:
    def is_in_building(outer_contours: list[np.ndarray], y: int, x: int):
        for contour in contours:
            return (cv2.pointPolygonTest(contour, (int(x), int(y)), measureDist=False) > 0
                    and primitive_mask[y, x] == 0)

    contours, _ = cv2.findContours(primitive_mask, mode=cv2.RETR_EXTERNAL,
                                   method=cv2.CHAIN_APPROX_SIMPLE)

    random_y = randrange(primitive_mask.shape[0])
    random_x = randrange(primitive_mask.shape[1])
    while not is_in_building(contours, random_y, random_x):
        random_y = randrange(primitive_mask.shape[0])
        random_x = randrange(primitive_mask.shape[1])

    return Point(random_x, random_y)
