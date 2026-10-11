"""
Базовые функции обработки изображений для практической работы №1.

Изображения: np.ndarray (H, W, 3), BGR, uint8.
Точки: (y, x).
Используются только базовые операции numpy. OpenCV применяется только
для imread/imwrite (в этом модуле cv2 вообще не импортируется).
"""

import random
from typing import List, Tuple, Set, Dict, Any

import numpy as np

Point = Tuple[int, int]


# ============================================================
# Базовые матричные операции (реализованы вручную)
# ============================================================

def _to_grayscale(image: np.ndarray) -> np.ndarray:
    """BGR -> gray: 0.114*B + 0.587*G + 0.299*R."""
    b = image[..., 0].astype(np.float32)
    g = image[..., 1].astype(np.float32)
    r = image[..., 2].astype(np.float32)
    return 0.114 * b + 0.587 * g + 0.299 * r


def _convolve2d(gray: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """2D свёртка произвольным ядром (edge-padding)."""
    kh, kw = kernel.shape
    ph, pw = kh // 2, kw // 2
    padded = np.pad(gray, ((ph, ph), (pw, pw)), mode='edge')
    h, w = gray.shape
    out = np.zeros((h, w), dtype=np.float32)
    for i in range(kh):
        for j in range(kw):
            out += kernel[i, j] * padded[i:i + h, j:j + w]
    return out


def _dilate(mask: np.ndarray, thickness: int) -> np.ndarray:
    """Утолщение маски. thickness — желаемый размер ядра (нечётное)."""
    if thickness <= 1:
        return mask.copy()
    r = (thickness - 1) // 2
    h, w = mask.shape
    padded = np.pad(mask, ((r, r), (r, r)), mode='constant')
    out = np.zeros((h, w), dtype=mask.dtype)
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            out = np.maximum(out, padded[r + dy:r + dy + h, r + dx:r + dx + w])
    return out


def _gaussian_kernel_1d(sigma: float) -> np.ndarray:
    radius = max(1, int(round(3 * sigma)))
    x = np.arange(-radius, radius + 1, dtype=np.float32)
    k = np.exp(-(x ** 2) / (2.0 * sigma ** 2))
    return k / k.sum()


def _gaussian_blur(gray: np.ndarray, sigma: float) -> np.ndarray:
    """Разделимое гауссово размытие (строки, потом столбцы)."""
    k = _gaussian_kernel_1d(sigma)
    r = len(k) // 2
    h, w = gray.shape

    padded = np.pad(gray, ((r, r), (0, 0)), mode='edge')
    tmp = np.zeros_like(gray, dtype=np.float32)
    for i, kv in enumerate(k):
        tmp += kv * padded[i:i + h, :]

    padded = np.pad(tmp, ((0, 0), (r, r)), mode='edge')
    out = np.zeros_like(gray, dtype=np.float32)
    for i, kv in enumerate(k):
        out += kv * padded[:, i:i + w]
    return out


def _ellipse_mask(h: int, w: int) -> np.ndarray:
    """Бинарная маска эллипса: центр — центр изображения,
    полуоси = w/2 и h/2."""
    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0 # зачем?
    a, b = w / 2.0, h / 2.0
    ys = np.arange(h, dtype=np.float32)[:, None]
    xs = np.arange(w, dtype=np.float32)[None, :]
    inside = ((xs - cx) / a) ** 2 + ((ys - cy) / b) ** 2 <= 1.0 # зачем?
    return inside.astype(np.uint8) * 255


def _bresenham_line(p1: Point, p2: Point) -> List[Point]:
    """Целочисленный отрезок между p1 и p2 (алгоритм Брезенхэма)."""
    y1, x1 = int(p1[0]), int(p1[1])
    y2, x2 = int(p2[0]), int(p2[1])
    pts: List[Point] = []
    dy = abs(y2 - y1)
    dx = abs(x2 - x1)
    sy = 1 if y1 < y2 else -1
    sx = 1 if x1 < x2 else -1
    err = dx - dy
    while True:
        pts.append((y1, x1))
        if y1 == y2 and x1 == x2:
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x1 += sx
        if e2 < dx:
            err += dx
            y1 += sy
    return pts


def _stamp_square(mask: np.ndarray, cy: int, cx: int, thickness: int) -> None:
    """Штамп квадрата thickness x thickness в точке (cy, cx)."""
    h, w = mask.shape
    r = (thickness - 1) // 2
    y0, y1_ = max(0, cy - r), min(h, cy + r + 1)
    x0, x1_ = max(0, cx - r), min(w, cx + r + 1)
    if y0 < y1_ and x0 < x1_:
        mask[y0:y1_, x0:x1_] = 255


# ============================================================
# 5. Генератор рандомных точек
# ============================================================

def generate_random_points(image_shape: Tuple[int, ...], count: int) -> List[Point]:
    h, w = image_shape[:2]
    return [(random.randint(0, h - 1), random.randint(0, w - 1))
            for _ in range(count)]


# ============================================================
# 1. Белый шум
# ============================================================

def add_white_noise(image: np.ndarray, count: int) -> np.ndarray:
    result = image.copy()
    for (y, x) in generate_random_points(result.shape, count):
        result[y, x] = (255, 255, 255)
    return result


# ============================================================
# 2. Чёрный шум
# ============================================================

def add_black_noise(image: np.ndarray, count: int) -> np.ndarray:
    result = image.copy()
    for (y, x) in generate_random_points(result.shape, count):
        result[y, x] = (0, 0, 0)
    return result


# ============================================================
# 3. Соль и перец
# ============================================================

def add_salt_pepper_noise(image: np.ndarray, count: int) -> np.ndarray:
    salt_count = count // 2
    pepper_count = count - salt_count
    result = add_white_noise(image, salt_count)
    result = add_black_noise(result, pepper_count)
    return result


# ============================================================
# Понижение интенсивности в точке
# ============================================================

def reduce_intensity_at_point(image: np.ndarray, point: Point,
                              factor: float = 0.5) -> np.ndarray:
    result = image.copy()
    y, x = point
    result[y, x] = np.clip(result[y, x].astype(np.float32) * factor, 0, 255)\
        .astype(np.uint8)
    return result # удалить


# ============================================================
# 4. Понижение интенсивности
# ============================================================

def reduce_intensity(image: np.ndarray,
                               points: Set[Point],
                               factor: float = 0.5) -> np.ndarray:
    result = image.copy()
    for point in points:
        y, x = point
        result[y, x] = np.clip(result[y, x].astype(np.float32) * factor, 0, 255)\
            .astype(np.uint8)
    return result


# ============================================================
# 6. Эллиптическая маска
# ============================================================

def apply_ellipse_mask(image: np.ndarray,
                       color: Tuple[int, int, int] = (255, 255, 255)) -> np.ndarray:
    h, w = image.shape[:2]
    mask = _ellipse_mask(h, w)
    result = image.copy()
    result[mask == 0] = color #
    return result


# ============================================================
# 7. Разрастание региона
# ============================================================

from collections import deque


def grow_region(image: np.ndarray,
                center: Point,
                depth: int,
                tolerance: int = 0,
                connectivity: int = 8) -> Set[Point]:
    """
    Разрастание региона от точки `center` на `depth` порядков соседства.

    depth=0 -> только центр
    depth=1 -> центр + его соседи
    depth=2 -> плюс соседи соседей
    ...

    В регион попадают только пиксели того же цвета, что и центральный
    (с допуском `tolerance` по каждому каналу).
    `connectivity` = 4 или 8 — сколько соседей рассматривать.

    Возвращает множество точек (y, x).
    """
    h, w = image.shape[:2]
    cy, cx = center

    if not (0 <= cy < h and 0 <= cx < w):
        return set()

    seed_color = image[cy, cx].astype(np.int32)

    if connectivity == 8:
        neighbors = ((-1, -1), (-1, 0), (-1, 1),
                     (0, -1),           (0, 1),
                     (1, -1),  (1, 0),  (1, 1))
    else:
        neighbors = ((-1, 0), (1, 0), (0, -1), (0, 1))

    def same_color(y: int, x: int) -> bool:
        diff = np.abs(image[y, x].astype(np.int32) - seed_color)
        return int(diff.max()) <= tolerance

    region: Set[Point] = set()
    visited: Set[Point] = {(cy, cx)}
    frontier: deque = deque([(cy, cx)])

    # depth + 1, потому что depth=0 — это сам центр (нулевой порядок)
    for _ in range(depth + 1):
        if not frontier:
            break
        next_frontier: deque = deque()
        for (y, x) in frontier:
            region.add((y, x))
            for dy, dx in neighbors:
                ny, nx = y + dy, x + dx
                if (ny, nx) in visited:
                    continue
                if not (0 <= ny < h and 0 <= nx < w):
                    continue
                if not same_color(ny, nx):
                    continue
                visited.add((ny, nx))
                next_frontier.append((ny, nx))
        frontier = next_frontier

    return region



# ============================================================
# Функция возврата множества всех точек изображения
# ============================================================



def all_points(image: np.ndarray) -> Set[Point]:
    """Возвращает множество всех точек (y, x) изображения."""
    h, w = image.shape[:2]
    return {(y, x) for y in range(h) for x in range(w)}


# ============================================================
# 9. Таблица "тип плёнки -> дефект" и обработчики
# ============================================================

def _defect_darken(px: np.ndarray, factor: float) -> np.ndarray:
    return np.clip(px.astype(np.float32) * factor, 0, 255).astype(np.uint8)


def _defect_lighten(px: np.ndarray, delta: int) -> np.ndarray:
    return np.clip(px.astype(np.int32) + delta, 0, 255).astype(np.uint8)


def _defect_color_shift(px: np.ndarray, shift: Tuple[int, int, int]) -> np.ndarray:
    return np.clip(px.astype(np.int32) + np.array(shift), 0, 255).astype(np.uint8)


def _defect_desaturate(px: np.ndarray, percent: float) -> np.ndarray:
    b, g, r = float(px[0]), float(px[1]), float(px[2])
    gray = 0.114 * b + 0.587 * g + 0.299 * r
    return np.clip(px.astype(np.float32) +
                   (np.array([gray, gray, gray], dtype=np.float32) -
                    px.astype(np.float32)) * percent,
                   0, 255).astype(np.uint8)


DEFECT_HANDLERS = {
    'darken':      _defect_darken,
    'lighten':     _defect_lighten,
    'color_shift': _defect_color_shift,
    'desaturate':  _defect_desaturate,
}


FILM_DEFECT_TABLE: Dict[str, Dict[str, Any]] = {
    'infrared':    {'defect': 'color_shift', 'params': {'shift': (40, -20, -10)}},
    'old_film':    {'defect': 'darken',      'params': {'factor': 0.7}},
    'overexposed': {'defect': 'lighten',     'params': {'delta': 60}},
    'faded':       {'defect': 'desaturate',  'params': {'percent': 0.7}},
    'vintage':     {'defect': 'color_shift', 'params': {'shift': (10, 20, -30)}},
}


# ============================================================
# 8. Применение дефекта плёнки
# ============================================================

def apply_film_defect(image: np.ndarray, film_type: str,
                      region: Set[Point]) -> np.ndarray:
    if film_type not in FILM_DEFECT_TABLE:
        raise ValueError(f'Unknown film type: {film_type}')
    spec = FILM_DEFECT_TABLE[film_type]
    handler = DEFECT_HANDLERS[spec['defect']]
    params = spec['params']

    result = image.copy()
    for (y, x) in region:
        result[y, x] = handler(result[y, x], **params)
    return result


# ============================================================
# 10. Засеривание ("выцветание")
# ============================================================

def desaturate_region(image: np.ndarray, region: Set[Point],
                      percent: float) -> np.ndarray:
    result = image.copy().astype(np.float32)
    for (y, x) in region:
        px = result[y, x]
        gray = 0.114 * px[0] + 0.587 * px[1] + 0.299 * px[2]
        gray_px = np.array([gray, gray, gray], dtype=np.float32)
        result[y, x] = px + (gray_px - px) * percent
    return np.clip(result, 0, 255).astype(np.uint8)


# ============================================================
# 11. Генератор рандомных отрезков (царапин) + растеризация
# ============================================================

def generate_scratches(image_shape: Tuple[int, ...],
                       scratch_coefficient: float,
                       max_segments: int = 30,
                       max_thickness: int = 3
                       ) -> List[Tuple[Point, Point, int]]:
    h, w = image_shape[:2]
    n = max(1, int(max_segments * scratch_coefficient))
    scratches: List[Tuple[Point, Point, int]] = []
    for _ in range(n):
        p1 = (random.randint(0, h - 1), random.randint(0, w - 1))
        p2 = (random.randint(0, h - 1), random.randint(0, w - 1))
        thickness = random.randint(1, max_thickness)
        scratches.append((p1, p2, thickness))
    return scratches


def scratches_to_mask(image_shape: Tuple[int, ...],
                      scratches: List[Tuple[Point, Point, int]]) -> np.ndarray:
    h, w = image_shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    for (p1, p2, thickness) in scratches:
        for (y, x) in _bresenham_line(p1, p2):
            _stamp_square(mask, y, x, thickness)
    return mask


# ============================================================
# 12. Царапание
# ============================================================

def apply_scratches(image: np.ndarray,
                    scratch_coefficient: float,
                    scratch_color: Tuple[int, int, int] = (255, 255, 255),
                    intensity: float = 0.8) -> np.ndarray:
    scratches = generate_scratches(image.shape, scratch_coefficient)
    mask = scratches_to_mask(image.shape, scratches)
    mask_bool = mask > 0

    result = image.copy()
    color_layer = np.full_like(image, scratch_color)
    result[mask_bool] = (
        result[mask_bool].astype(np.float32) * (1 - intensity)
        + color_layer[mask_bool].astype(np.float32) * intensity
    ).astype(np.uint8)
    return result


# ============================================================
# 13. Выделение краёв (Sobel + гистерезис)
# ============================================================

_SOBEL_X = np.array([[-1, 0, 1],
                     [-2, 0, 2],
                     [-1, 0, 1]], dtype=np.float32)

_SOBEL_Y = np.array([[-1, -2, -1],
                     [ 0,  0,  0],
                     [ 1,  2,  1]], dtype=np.float32)


def detect_edges(image: np.ndarray,
                 thickness: int = 1,
                 low_threshold: int = 100,
                 high_threshold: int = 200) -> np.ndarray:
    gray = _to_grayscale(image)
    gx = _convolve2d(gray, _SOBEL_X)
    gy = _convolve2d(gray, _SOBEL_Y)
    mag = np.sqrt(gx ** 2 + gy ** 2)

    strong = (mag >= high_threshold)
    weak = (mag >= low_threshold) & ~strong

    edges = np.zeros_like(mag, dtype=np.uint8)
    edges[strong] = 255

    if weak.any() and strong.any():
        strong_dil = _dilate(edges, 3)
        edges[weak & (strong_dil > 0)] = 255

    if thickness > 1:
        edges = _dilate(edges, thickness)
    return edges


# ============================================================
# 14. Неонификация
# ============================================================

def neonify(image: np.ndarray,
            thickness: int = 3,
            glow_color: Tuple[int, int, int] = (255, 0, 255),
            glow_intensity: float = 0.8,
            blur_sigma: float = 7.0) -> np.ndarray:
    edges = detect_edges(image, thickness=thickness)
    glow = _gaussian_blur(edges.astype(np.float32), sigma=blur_sigma)
    glow = np.clip(glow, 0, 255)

    glow_colored = np.zeros_like(image)
    glow_colored[:] = glow_color

    alpha = np.clip((glow / 255.0) * glow_intensity, 0.0, 1.0)[..., None]

    darkened = image.astype(np.float32) * 0.3
    result = darkened * (1 - alpha) + glow_colored.astype(np.float32) * alpha
    result = np.clip(result, 0, 255).astype(np.uint8)

    result[edges > 0] = glow_color
    return result
