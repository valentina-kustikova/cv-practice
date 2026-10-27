"""
Базовые функции обработки изображений для практической работы №1.

Изображения: np.ndarray (H, W, 3), BGR, uint8.
Точки: (y, x).
Используются только базовые операции numpy. OpenCV применяется только
для imread/imwrite (в этом модуле cv2 вообще не импортируется).
"""

import random
import math
from collections import deque
from typing import List, Tuple, Set, Dict, Any, Optional

import numpy as np

Point = Tuple[int, int]


# ============================================================
# Базовые матричные операции
# ============================================================

def _to_grayscale(image: np.ndarray) -> np.ndarray:
    b = image[..., 0].astype(np.float32)
    g = image[..., 1].astype(np.float32)
    r = image[..., 2].astype(np.float32)
    return 0.114 * b + 0.587 * g + 0.299 * r


def _convolve2d(gray: np.ndarray, kernel: np.ndarray) -> np.ndarray:
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
    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
    a, b = w / 2.0, h / 2.0
    ys = np.arange(h, dtype=np.float32)[:, None]
    xs = np.arange(w, dtype=np.float32)[None, :]
    inside = ((xs - cx) / a) ** 2 + ((ys - cy) / b) ** 2 <= 1.0
    return inside.astype(np.uint8) * 255


def _bresenham_line(p1: Point, p2: Point) -> List[Point]:
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
    h, w = mask.shape
    r = (thickness - 1) // 2
    y0, y1_ = max(0, cy - r), min(h, cy + r + 1)
    x0, x1_ = max(0, cx - r), min(w, cx + r + 1)
    if y0 < y1_ and x0 < x1_:
        mask[y0:y1_, x0:x1_] = 255


# ============================================================
# 5. Генератор случайных точек
# ============================================================

def generate_random_points(image_shape: Tuple[int, ...], count: int) -> List[Point]:
    h, w = image_shape[:2]
    return [(random.randint(0, h - 1), random.randint(0, w - 1))
            for _ in range(count)]


# ============================================================
# 1-3. Шумы
# ============================================================

def add_white_noise(image: np.ndarray, count: int) -> np.ndarray:
    result = image.copy()
    for (y, x) in generate_random_points(result.shape, count):
        result[y, x] = (255, 255, 255)
    return result


def add_black_noise(image: np.ndarray, count: int) -> np.ndarray:
    result = image.copy()
    for (y, x) in generate_random_points(result.shape, count):
        result[y, x] = (0, 0, 0)
    return result


def add_salt_pepper_noise(image: np.ndarray, count: int) -> np.ndarray:
    salt_count = count // 2
    pepper_count = count - salt_count
    result = add_white_noise(image, salt_count)
    result = add_black_noise(result, pepper_count)
    return result


# ============================================================
# 4. Понижение интенсивности
# ============================================================

def reduce_intensity_at_point(image: np.ndarray, point: Point,
                              factor: float = 0.5) -> np.ndarray:
    result = image.copy()
    y, x = point
    result[y, x] = np.clip(result[y, x].astype(np.float32) * factor, 0, 255)\
        .astype(np.uint8)
    return result


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
    result[mask == 0] = color
    return result


# ============================================================
# 7. Разрастание региона
# ============================================================

def grow_region(image: np.ndarray,
                center: Point,
                depth: int,
                tolerance: int = 0,
                connectivity: int = 8) -> Set[Point]:
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


def all_points(image: np.ndarray) -> Set[Point]:
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
# 10. Засеривание
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
# 11-12. Царапины
# ============================================================

def _fill_rhombus(mask: np.ndarray,
                  p1: Point, p2: Point,
                  half_diag_fraction: float = 0.02) -> None:
    """
    Заполняет в mask ромб с вершинами p1, p3, p2, p4.

    - p1, p2 — концы отрезка (главная диагональ).
    - p3, p4 — на смещении от середины:
          x_offset = ±L * half_diag_fraction * sin(angle)
          y_offset = ±L * half_diag_fraction * cos(angle)
      где sin(angle) = dy/L, cos(angle) = dx/L, L — длина отрезка.

    Заливаются точки из bbox{p1,p2,p3,p4}, лежащие внутри фигуры,
    ограниченной прямыми p1-p3, p3-p2, p2-p4, p4-p1.
    """
    h, w = mask.shape[:2]
    y1, x1 = int(p1[0]), int(p1[1])
    y2, x2 = int(p2[0]), int(p2[1])

    dy = y2 - y1
    dx = x2 - x1
    length = math.hypot(dy, dx)

    if length == 0:
        if 0 <= y1 < h and 0 <= x1 < w:
            mask[y1, x1] = 255
        return

    # sin/cos угла, под которым лежит отрезок (от оси x)
    sin_a = dy / length
    cos_a = dx / length

    # Середина отрезка
    my = (y1 + y2) / 2.0
    mx = (x1 + x2) / 2.0

    # Смещение для p3, p4
    offset = length * half_diag_fraction
    x_off = offset * sin_a
    y_off = offset * cos_a

    # p3, p4 в координатах (y, x)
    py3 = my + y_off
    px3 = mx + x_off
    py4 = my - y_off
    px4 = mx - x_off

    # Вырожденный случай: все 4 точки оказались на одной прямой.
    # Такое бывает, например, для строго диагональных отрезков (dy == ±dx).
    # В этом случае формула даёт не ромб, а отрезок — заливаем его.
    cross_3 = (y2 - y1) * (px3 - x1) - (x2 - x1) * (py3 - y1)
    cross_4 = (y2 - y1) * (px4 - x1) - (x2 - x1) * (py4 - y1)
    if abs(cross_3) < 1e-6 and abs(cross_4) < 1e-6:
        steps = max(abs(dy), abs(dx), 1)
        for i in range(steps + 1):
            t = i / steps
            yy = int(round(y1 + t * dy))
            xx = int(round(x1 + t * dx))
            if 0 <= yy < h and 0 <= xx < w:
                mask[yy, xx] = 255
        return

    # Bbox, содержащий все 4 вершины
    ys_all = (y1, y2, py3, py4)
    xs_all = (x1, x2, px3, px4)
    y_min = max(0, int(math.floor(min(ys_all))) - 1)
    y_max = min(h - 1, int(math.ceil(max(ys_all))) + 1)
    x_min = max(0, int(math.floor(min(xs_all))) - 1)
    x_max = min(w - 1, int(math.ceil(max(xs_all))) + 1)

    # Центр ромба = середина p1-p2 = середина p3-p4
    cy, cx = my, mx

    # Знаки cross-произведений центра относительно каждой из 4 сторон
    # задают «внутреннюю» сторону.
    ref_1 = (px3 - x1) * (cy - y1) - (py3 - y1) * (cx - x1)
    ref_2 = (x2 - px3) * (cy - py3) - (y2 - py3) * (cx - px3)
    ref_3 = (px4 - x2) * (cy - y2) - (py4 - y2) * (cx - x2)
    ref_4 = (x1 - px4) * (cy - py4) - (y1 - py4) * (cx - px4)

    for y in range(y_min, y_max + 1):
        for x in range(x_min, x_max + 1):
            s1 = (px3 - x1) * (y - y1) - (py3 - y1) * (x - x1)
            s2 = (x2 - px3) * (y - py3) - (y2 - py3) * (x - px3)
            s3 = (px4 - x2) * (y - y2) - (py4 - y2) * (x - x2)
            s4 = (x1 - px4) * (y - py4) - (y1 - py4) * (x - px4)
            if (s1 * ref_1 >= 0 and s2 * ref_2 >= 0
                    and s3 * ref_3 >= 0 and s4 * ref_4 >= 0):
                mask[y, x] = 255


def generate_scratches(image_shape: Tuple[int, ...],
                       scratch_coefficient: float,
                       max_segments: int = 30,
                       max_thickness: int = 3,
                       max_length: float = -1
                       ) -> List[Tuple[Point, Point, int]]:
    """
    Генерирует n = max_segments * scratch_coefficient царапин.

    Если задан max_length (в пикселях):
      генерируются случайные кандидаты, из них отбираются только те,
      чья длина не превышает max_length; процесс повторяется, пока не
      наберётся n подходящих царапин (или пока не будет исчерпан лимит
      попыток — тогда возвращается меньше n).

    max_length=None или <= 0 — без ограничения (прежнее поведение).
    """
    h, w = image_shape[:2]
    n = max(1, int(max_segments * scratch_coefficient))

    # Без ограничения длины — генерируем n штук сразу
    if max_length <= 0:
        max_length = max_segments + 5

    scratches: List[Tuple[Point, Point, int]] = []
    attempts = 0
    max_attempts = n * 100  # страховка от бесконечного цикла

    while len(scratches) < n and attempts < max_attempts:
        # Батч кандидатов размером «сколько ещё не хватает»
        remaining = n - len(scratches)
        batch: List[Tuple[Point, Point, int]] = []

        for _ in range(remaining):
            if attempts >= max_attempts:
                break
            p1 = (random.randint(0, h - 1), random.randint(0, w - 1))
            p2 = (random.randint(0, h - 1), random.randint(0, w - 1))
            attempts += 1

            length = p2[0] - p1[0] + p2[1] - p1[1] # манхэттен
            if length <= max_length:
                thickness = random.randint(1, max_thickness)
                batch.append((p1, p2, thickness))

        # Отбираем из батча столько, сколько ещё нужно
        need = n - len(scratches)
        scratches.extend(batch[:need])

    return scratches


def scratches_to_mask(image_shape: Tuple[int, ...],
                      scratches: List[Tuple[Point, Point, int]],
                      half_diag_fraction: float = 0.02) -> np.ndarray:
    h, w = image_shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    for entry in scratches:
        p1, p2 = entry[0], entry[1]
        _fill_rhombus(mask, p1, p2, half_diag_fraction)
    return mask


def apply_scratches(image: np.ndarray,
                    scratch_coefficient: float,
                    scratch_color: Tuple[int, int, int] = (255, 255, 255),
                    intensity: float = 0.5,
                    max_segments: int = 20,
                    max_thickness: int = 3,
                    half_diag_fraction: float = 0.02) -> np.ndarray:
    scratches = generate_scratches(image.shape, scratch_coefficient,
                                   max_segments, max_thickness)
    mask = scratches_to_mask(image.shape, scratches, half_diag_fraction)
    mask_bool = mask > 0

    result = image.copy()
    color_layer = np.full_like(image, scratch_color)
    result[mask_bool] = (
        result[mask_bool].astype(np.float32) * (1 - intensity)
        + color_layer[mask_bool].astype(np.float32) * intensity
    ).astype(np.uint8)
    return result


# ============================================================
# 13. Края (Sobel + гистерезис)
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


# ============================================================
# 15. Оцу
# ============================================================

def otsu_threshold(gray: np.ndarray) -> int:
    hist = np.bincount(gray.ravel().astype(np.uint8), minlength=256)\
             .astype(np.float64)
    total = hist.sum()
    if total == 0:
        return 0

    levels = np.arange(256, dtype=np.float64)
    cum_count = np.cumsum(hist)
    cum_sum = np.cumsum(hist * levels)

    w0 = cum_count / total
    w1 = 1.0 - w0

    mu0 = np.zeros(256, dtype=np.float64)
    mu1 = np.zeros(256, dtype=np.float64)
    valid0 = cum_count > 0
    valid1 = cum_count < total
    mu0[valid0] = cum_sum[valid0] / cum_count[valid0]
    mu1[valid1] = (cum_sum[-1] - cum_sum[valid1]) / (total - cum_count[valid1])

    sigma_b = w0 * w1 * (mu0 - mu1) ** 2
    sigma_b[~np.isfinite(sigma_b)] = 0.0
    return int(np.argmax(sigma_b))


def otsu_binarize(image: np.ndarray, invert: bool = False) -> np.ndarray:
    if image.ndim == 3:
        gray = _to_grayscale(image).astype(np.uint8)
    else:
        gray = image.astype(np.uint8)

    t = otsu_threshold(gray)
    binary = np.where(gray > t, 255, 0).astype(np.uint8)

    if invert:
        binary = 255 - binary
    return binary


# ============================================================
# 16-17. edxdy — регионы между вложенными эллипсами
# ============================================================

def _ellipse_boundary(cx: float, cy: float,
                      a: float, b: float,
                      W: int, H: int) -> Set[Point]:
    if a <= 0 or b <= 0:
        return set()

    perimeter = math.pi * (
        3 * (a + b) - math.sqrt(max(0.0, (3 * a + b) * (a + 3 * b)))
    )
    n_samples = max(8, int(perimeter * 4))

    pts: Set[Point] = set()
    for k in range(n_samples):
        theta = 2.0 * math.pi * k / n_samples
        x = int(round(cx + a * math.cos(theta)))
        y = int(round(cy + b * math.sin(theta)))
        if 0 <= y < H and 0 <= x < W:
            pts.add((y, x))
    return pts


def edxdy(image: np.ndarray,
          center: Point,
          h_semi: float, v_semi: float,
          dx: float, dy: float,
          max_rings: int = 500) -> List[Set[Point]]:
    H, W = image.shape[:2]
    cy, cx = center

    if dx <= 0 or dy <= 0:
        raise ValueError('dx and dy must be positive')
    if h_semi <= 0 or v_semi <= 0:
        raise ValueError('semi-axes must be positive')

    ellipses: List[Tuple[float, float]] = []
    i = 0
    while i < max_rings:
        a = h_semi + i * dx
        b = v_semi + i * dy
        ellipses.append((a, b))

        max_h = max(abs(cx - 0.0), abs(cx - (W - 1)))
        max_v = max(abs(cy - 0.0), abs(cy - (H - 1)))
        if (max_h / a) ** 2 + (max_v / b) ** 2 <= 1.0:
            break
        i += 1

    N = len(ellipses)
    if N == 0:
        return []

    ys = np.arange(H, dtype=np.float32)[:, None]
    xs = np.arange(W, dtype=np.float32)[None, :]

    result: List[Set[Point]] = []
    prev_inside = None

    for i, (a, b) in enumerate(ellipses):
        e = ((xs - cx) / a) ** 2 + ((ys - cy) / b) ** 2
        cur_inside = e <= 1.0

        if i == 0:
            region_mask = cur_inside
        else:
            region_mask = cur_inside & ~prev_inside

        ys_r, xs_r = np.where(region_mask)
        result.append(set(zip(ys_r.tolist(), xs_r.tolist())))
        prev_inside = cur_inside

    return result


def edxdy_masks(image: np.ndarray,
                center: Point,
                h_semi: float, v_semi: float,
                dx: float, dy: float,
                max_rings: int = 500) -> List[np.ndarray]:
    H, W = image.shape[:2]
    cy, cx = center

    if dx <= 0 or dy <= 0:
        raise ValueError('dx and dy must be positive')
    if h_semi <= 0 or v_semi <= 0:
        raise ValueError('semi-axes must be positive')

    ellipses: List[Tuple[float, float]] = []
    i = 0
    while i < max_rings:
        a = h_semi + i * dx
        b = v_semi + i * dy
        ellipses.append((a, b))
        max_h = max(abs(cx - 0.0), abs(cx - (W - 1)))
        max_v = max(abs(cy - 0.0), abs(cy - (H - 1)))
        if (max_h / a) ** 2 + (max_v / b) ** 2 <= 1.0:
            break
        i += 1

    N = len(ellipses)
    if N == 0:
        return []

    ys = np.arange(H, dtype=np.float32)[:, None]
    xs = np.arange(W, dtype=np.float32)[None, :]

    result: List[np.ndarray] = []
    prev_inside = None

    for i, (a, b) in enumerate(ellipses):
        e = ((xs - cx) / a) ** 2 + ((ys - cy) / b) ** 2
        cur_inside = e <= 1.0

        if i == 0:
            result.append(cur_inside.copy())
        else:
            result.append(cur_inside & ~prev_inside)

        prev_inside = cur_inside

    return result


# ============================================================
# 18-20. Виньетки
# ============================================================

def _apply_vignette_rings_sets(image: np.ndarray,
                               rings: List[Set[Point]],
                               brighten: bool,
                               strength: float) -> np.ndarray:
    N = len(rings)
    if N <= 1:
        return image.copy()

    result = image.astype(np.float32)
    for i, ring in enumerate(rings):
        amount = (i / (N - 1)) * strength
        amount = min(max(amount, 0.0), 1.0)
        for (y, x) in ring:
            px = result[y, x]
            if brighten:
                result[y, x] = px + (255.0 - px) * amount
            else:
                result[y, x] = px * (1.0 - amount)
    return np.clip(result, 0, 255).astype(np.uint8)


def _apply_vignette_rings_masks(image: np.ndarray,
                                masks: List[np.ndarray],
                                brighten: bool,
                                strength: float) -> np.ndarray:
    N = len(masks)
    if N <= 1:
        return image.copy()

    result = image.astype(np.float32)
    for i, mask in enumerate(masks):
        amount = (i / (N - 1)) * strength
        amount = min(max(amount, 0.0), 1.0)
        if not mask.any():
            continue
        px = result[mask]
        if brighten:
            result[mask] = px + (255.0 - px) * amount
        else:
            result[mask] = px * (1.0 - amount)
    return np.clip(result, 0, 255).astype(np.uint8)


def _apply_vignette_random_sets(image: np.ndarray,
                                rings: List[Set[Point]],
                                brighten: bool,
                                interval_len: float,
                                strength: float) -> np.ndarray:
    N = len(rings)
    if N <= 1:
        return image.copy()

    L = float(interval_len)
    if not 0.0 < L <= 1.0:
        raise ValueError('interval_len must be in (0, 1]')

    result = image.astype(np.float32)
    for i, ring in enumerate(rings):
        t = i / (N - 1)
        lo = t * (1.0 - L)
        hi = lo + L
        for (y, x) in ring:
            r = random.uniform(lo, hi) * strength
            r = min(max(r, 0.0), 1.0)
            px = result[y, x]
            if brighten:
                result[y, x] = px + (255.0 - px) * r
            else:
                result[y, x] = px * (1.0 - r)
    return np.clip(result, 0, 255).astype(np.uint8)


def _apply_vignette_random_masks(image: np.ndarray,
                                 masks: List[np.ndarray],
                                 brighten: bool,
                                 interval_len: float,
                                 strength: float) -> np.ndarray:
    N = len(masks)
    if N <= 1:
        return image.copy()

    L = float(interval_len)
    if not 0.0 < L <= 1.0:
        raise ValueError('interval_len must be in (0, 1]')

    result = image.astype(np.float32)
    for i, mask in enumerate(masks):
        if not mask.any():
            continue
        t = i / (N - 1)
        lo = t * (1.0 - L)
        hi = lo + L

        n = int(mask.sum())
        r = np.random.uniform(lo, hi, size=n) * strength
        r = np.clip(r, 0.0, 1.0)

        px = result[mask]
        if brighten:
            result[mask] = px + (255.0 - px) * r[:, None]
        else:
            result[mask] = px * (1.0 - r[:, None])
    return np.clip(result, 0, 255).astype(np.uint8)


# Публичные виньетки — с центром, работают поверх edxdy

def vignette_rings(image: np.ndarray,
                   center: Point,
                   h_semi: float, v_semi: float,
                   dx: float, dy: float,
                   brighten: bool,
                   strength: float = 1.0) -> np.ndarray:
    rings = edxdy(image, center, h_semi, v_semi, dx, dy)
    return _apply_vignette_rings_sets(image, rings, brighten, strength)


def vignette_rings_masks(image: np.ndarray,
                         center: Point,
                         h_semi: float, v_semi: float,
                         dx: float, dy: float,
                         brighten: bool,
                         strength: float = 1.0) -> np.ndarray:
    masks = edxdy_masks(image, center, h_semi, v_semi, dx, dy)
    return _apply_vignette_rings_masks(image, masks, brighten, strength)


def vignette_random_rings(image: np.ndarray,
                          center: Point,
                          h_semi: float, v_semi: float,
                          dx: float, dy: float,
                          brighten: bool,
                          interval_len: float = 0.2,
                          strength: float = 1.0) -> np.ndarray:
    rings = edxdy(image, center, h_semi, v_semi, dx, dy)
    return _apply_vignette_random_sets(image, rings, brighten,
                                       interval_len, strength)


def vignette_random_rings_masks(image: np.ndarray,
                                center: Point,
                                h_semi: float, v_semi: float,
                                dx: float, dy: float,
                                brighten: bool,
                                interval_len: float = 0.2,
                                strength: float = 1.0) -> np.ndarray:
    masks = edxdy_masks(image, center, h_semi, v_semi, dx, dy)
    return _apply_vignette_random_masks(image, masks, brighten,
                                        interval_len, strength)


def vignette_smooth(image: np.ndarray,
                    h_semi: float, v_semi: float,
                    brighten: bool,
                    strength: float = 1.0) -> np.ndarray:
    H, W = image.shape[:2]
    cy, cx = (H - 1) / 2.0, (W - 1) / 2.0

    ys, xs = np.mgrid[0:H, 0:W].astype(np.float32)
    u = (xs - cx) / h_semi
    v = (ys - cy) / v_semi
    r = np.sqrt(u * u + v * v)

    r_max = float(r.max())
    if r_max <= 1.0:
        return image.copy()

    amount = np.clip((r - 1.0) / (r_max - 1.0), 0.0, 1.0) * strength
    amount = np.minimum(amount, 1.0)[..., None]

    result = image.astype(np.float32)
    if brighten:
        result = result + (255.0 - result) * amount
    else:
        result = result * (1.0 - amount)

    return np.clip(result, 0, 255).astype(np.uint8)


# ============================================================
# 21. Сепия
# ============================================================

_SEPIA_MATRIX = np.array([
    [0.393, 0.769, 0.189],
    [0.349, 0.686, 0.168],
    [0.272, 0.534, 0.131],
], dtype=np.float32)


def apply_sepia(image: np.ndarray,
                strength: float = 1.0) -> np.ndarray:
    r = image[..., 2].astype(np.float32)
    g = image[..., 1].astype(np.float32)
    b = image[..., 0].astype(np.float32)

    new_r = _SEPIA_MATRIX[0, 0] * r + _SEPIA_MATRIX[0, 1] * g + _SEPIA_MATRIX[0, 2] * b
    new_g = _SEPIA_MATRIX[1, 0] * r + _SEPIA_MATRIX[1, 1] * g + _SEPIA_MATRIX[1, 2] * b
    new_b = _SEPIA_MATRIX[2, 0] * r + _SEPIA_MATRIX[2, 1] * g + _SEPIA_MATRIX[2, 2] * b

    sepia = np.stack([new_b, new_g, new_r], axis=-1)
    sepia = np.clip(sepia, 0, 255)

    if strength >= 1.0:
        return sepia.astype(np.uint8)
    if strength <= 0.0:
        return image.copy()

    blended = image.astype(np.float32) * (1.0 - strength) + sepia * strength
    return np.clip(blended, 0, 255).astype(np.uint8)


# ============================================================
# 22. Состаривание фото: сепия -> царапины -> vignette_random_rings
# ============================================================

def apply_aging(image: np.ndarray,
                center: Point,
                sepia_strength: float = 1.0,
                scratch_coefficient: float = 0.8,
                scratch_color: Tuple[int, int, int] = (255, 255, 255),
                scratch_intensity: float = 0.6,
                max_segments: int = 10,
                max_thickness: int = 3,
                half_diag_fraction: float = 0.02,
                vignette_h_semi: Optional[float] = None,
                vignette_v_semi: Optional[float] = None,
                vignette_dx: float = 10.0,
                vignette_dy: float = 10.0,
                vignette_brighten: bool = False,
                vignette_interval_len: float = 0.2,
                vignette_strength: float = 1.0) -> np.ndarray:
    """
    Состаривание фото: последовательно применяет
        1) сепию,
        2) царапины,
        3) виньетку со случайными коэффициентами в кольцах.

    center — явный центр для виньетки.
    Если полуоси виньетки не заданы, берутся w/4 и h/4.
    """
    H, W = image.shape[:2]
    if vignette_h_semi is None:
        vignette_h_semi = W / 4.0
    if vignette_v_semi is None:
        vignette_v_semi = H / 4.0

    result = apply_sepia(image, strength=sepia_strength)
    result = apply_scratches(result,
                             scratch_coefficient=scratch_coefficient,
                             scratch_color=scratch_color,
                             intensity=scratch_intensity,
                             max_segments=max_segments,
                             max_thickness=max_thickness,
                             half_diag_fraction=half_diag_fraction)
    result = vignette_random_rings(result,
                                   center=center,
                                   h_semi=vignette_h_semi,
                                   v_semi=vignette_v_semi,
                                   dx=vignette_dx,
                                   dy=vignette_dy,
                                   brighten=vignette_brighten,
                                   interval_len=vignette_interval_len,
                                   strength=vignette_strength)
    return result
