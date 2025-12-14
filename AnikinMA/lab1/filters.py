import numpy as np
import math

def _ensure_bgr(img: np.ndarray) -> np.ndarray:
    if img is None:
        raise ValueError("img is None")
    if img.ndim == 2:
        return np.stack([img, img, img], axis=-1).astype(np.uint8)
    if img.ndim == 3 and img.shape[2] >= 3:
        return img[:, :, :3].astype(np.uint8, copy=False)
    raise ValueError(f"Unsupported image shape: {img.shape}")

def _to_float(img: np.ndarray) -> np.ndarray:
    return img.astype(np.float32, copy=False)

def _to_u8(img: np.ndarray) -> np.ndarray:
    return np.clip(img, 0, 255).astype(np.uint8)

def _resize_bilinear_float(img_f: np.ndarray, new_w: int, new_h: int) -> np.ndarray:
    H, W, C = img_f.shape
    new_w = int(new_w)
    new_h = int(new_h)
    if new_w <= 0 or new_h <= 0:
        raise ValueError("new_w/new_h must be positive")

    ys = (np.arange(new_h, dtype=np.float32) + 0.5) * (H / new_h) - 0.5
    xs = (np.arange(new_w, dtype=np.float32) + 0.5) * (W / new_w) - 0.5
    Y, X = np.meshgrid(ys, xs, indexing="ij")

    Y = np.clip(Y, 0, H - 1.001)
    X = np.clip(X, 0, W - 1.001)

    y0 = np.floor(Y).astype(np.int32)
    x0 = np.floor(X).astype(np.int32)
    y1 = np.clip(y0 + 1, 0, H - 1)
    x1 = np.clip(x0 + 1, 0, W - 1)

    wy = (Y - y0)[..., None]
    wx = (X - x0)[..., None]

    Ia = img_f[y0, x0]
    Ib = img_f[y0, x1]
    Ic = img_f[y1, x0]
    Id = img_f[y1, x1]

    wa = (1 - wy) * (1 - wx)
    wb = (1 - wy) * wx
    wc = wy * (1 - wx)
    wd = wy * wx

    return Ia * wa + Ib * wb + Ic * wc + Id * wd


def _resize_nearest(img: np.ndarray, new_w: int, new_h: int) -> np.ndarray:
    img = _ensure_bgr(img)
    H, W, _ = img.shape
    new_w = int(new_w)
    new_h = int(new_h)

    ys = (np.arange(new_h, dtype=np.float32) + 0.5) * (H / new_h) - 0.5
    xs = (np.arange(new_w, dtype=np.float32) + 0.5) * (W / new_w) - 0.5
    Y, X = np.meshgrid(ys, xs, indexing="ij")

    yn = np.clip(np.rint(Y).astype(np.int32), 0, H - 1)
    xn = np.clip(np.rint(X).astype(np.int32), 0, W - 1)
    return img[yn, xn]


def _point_in_poly_mask(H: int, W: int, poly_xy: np.ndarray) -> np.ndarray:
    poly = poly_xy.astype(np.float32)
    x = poly[:, 0]
    y = poly[:, 1]
    n = len(poly)

    Y, X = np.mgrid[0:H, 0:W].astype(np.float32)

    inside = np.zeros((H, W), dtype=bool)

    j = n - 1
    for i in range(n):
        xi, yi = x[i], y[i]
        xj, yj = x[j], y[j]

        intersect = ((yi > Y) != (yj > Y)) & (
            X < (xj - xi) * (Y - yi) / ((yj - yi) + 1e-12) + xi
        )
        inside ^= intersect
        j = i

    return inside

def change_resolution(img: np.ndarray, new_width: int, new_height: int) -> np.ndarray:
    img = _ensure_bgr(img)
    f = _to_float(img)
    out_f = _resize_bilinear_float(f, int(new_width), int(new_height))
    return _to_u8(out_f)


def apply_sepia(img: np.ndarray) -> np.ndarray:
    img = _ensure_bgr(img)
    f = _to_float(img)

    B = f[:, :, 0]
    G = f[:, :, 1]
    R = f[:, :, 2]

    Rp = 0.393 * R + 0.769 * G + 0.189 * B
    Gp = 0.349 * R + 0.686 * G + 0.168 * B
    Bp = 0.272 * R + 0.534 * G + 0.131 * B

    out = np.stack([Bp, Gp, Rp], axis=-1)
    return _to_u8(out)

def apply_vignette(img: np.ndarray, sigma: float = 200.0) -> np.ndarray:
    img = _ensure_bgr(img)
    f = _to_float(img)
    H, W, _ = f.shape

    cy = (H - 1) / 2.0
    cx = (W - 1) / 2.0

    Y, X = np.mgrid[0:H, 0:W].astype(np.float32)
    dy = Y - cy
    dx = X - cx
    r2 = dx * dx + dy * dy

    sigma = float(sigma)
    sigma = max(1.0, sigma)

    mask = np.exp(-r2 / (2.0 * sigma * sigma)).astype(np.float32)  # 0..1

    out = f * mask[..., None]
    return _to_u8(out)

def pixelate_region(img: np.ndarray, x: int, y: int, w: int, h: int, block_size: int) -> np.ndarray:
    img = _ensure_bgr(img)
    out = img.copy()
    H, W, _ = out.shape

    x = int(x); y = int(y); w = int(w); h = int(h)
    bs = max(1, int(block_size))

    x0 = max(0, x)
    y0 = max(0, y)
    x1 = min(W, x + w)
    y1 = min(H, y + h)
    if x0 >= x1 or y0 >= y1:
        return out

    region = out[y0:y1, x0:x1]
    rh, rw = region.shape[:2]

    for i in range(0, rh, bs):
        for j in range(0, rw, bs):
            block = region[i:i+bs, j:j+bs]
            if block.size == 0:
                continue
            avg = block.mean(axis=(0, 1))
            region[i:i+bs, j:j+bs] = avg.astype(np.uint8)

    return out

def add_rect_frame(img: np.ndarray, frame_width: int, frame_color=(0, 0, 0)) -> np.ndarray:
    img = _ensure_bgr(img)
    out = img.copy()
    H, W, _ = out.shape

    fw = max(1, int(frame_width))
    b, g, r = [int(v) for v in frame_color]

    out[:fw, :, :] = (b, g, r)
    out[-fw:, :, :] = (b, g, r)
    out[:, :fw, :] = (b, g, r)
    out[:, -fw:, :] = (b, g, r)
    return out

def add_shape_frame(img: np.ndarray, shape: str = "ellipse", frame_color=(0, 0, 0)) -> np.ndarray:
    img = _ensure_bgr(img)
    out = img.copy()
    H, W, _ = out.shape
    bgr = np.array(frame_color, dtype=np.uint8).reshape(1, 1, 3)

    Y, X = np.mgrid[0:H, 0:W].astype(np.float32)
    cx = (W - 1) / 2.0
    cy = (H - 1) / 2.0

    shape = shape.lower().strip()

    if shape == "ellipse":
        a = (W - 1) / 2.0
        b = (H - 1) / 2.0
        inside = ((X - cx) / (a + 1e-12)) ** 2 + ((Y - cy) / (b + 1e-12)) ** 2 <= 1.0

    elif shape == "circle":
        r = min((W - 1), (H - 1)) / 2.0
        inside = (X - cx) ** 2 + (Y - cy) ** 2 <= r * r

    elif shape == "diamond":
        a = (W - 1) / 2.0
        b = (H - 1) / 2.0
        inside = (np.abs(X - cx) / (a + 1e-12) + np.abs(Y - cy) / (b + 1e-12)) <= 1.0

    elif shape == "star":
        num = 5
        R = 0.45 * min(W, H)
        r = 0.5 * R
        pts = []
        angle0 = -90.0
        for i in range(num * 2):
            ang = math.radians(angle0 + i * (360.0 / (num * 2)))
            rad = R if (i % 2 == 0) else r
            px = cx + rad * math.cos(ang)
            py = cy + rad * math.sin(ang)
            pts.append([px, py])
        pts = np.array(pts, dtype=np.float32)
        inside = _point_in_poly_mask(H, W, pts)

    else:
        return out

    out[~inside] = bgr
    return out

def add_lens_flare(img: np.ndarray) -> np.ndarray:
    img = _ensure_bgr(img)
    f = _to_float(img)
    H, W, _ = f.shape

    Y, X = np.mgrid[0:H, 0:W].astype(np.float32)

    rng = np.random.default_rng()
    spots = rng.integers(3, 7)
    flare = np.zeros((H, W, 3), dtype=np.float32)

    for i in range(spots):
        cx = float(rng.integers(0, W))
        cy = float(rng.integers(0, H))
        radius = float(rng.integers(max(8, min(H, W)//50), max(25, min(H, W)//10)))
        sigma2 = (0.6 * radius) ** 2

        d2 = (X - cx) ** 2 + (Y - cy) ** 2
        g = np.exp(-d2 / (2.0 * sigma2))

        col = np.array([0.9, 0.95, 1.0], dtype=np.float32)
        amp = float(rng.uniform(60, 180))
        flare += g[..., None] * col[None, None, :] * amp

    out = f + flare
    return _to_u8(out)

def add_watercolor_texture(img: np.ndarray) -> np.ndarray:
    img = _ensure_bgr(img)
    f = _to_float(img)
    H, W, _ = f.shape

    rng = np.random.default_rng()

    tex = np.zeros((H, W), dtype=np.float32)

    levels = [
        (max(2, W // 32), max(2, H // 32), 18.0),
        (max(2, W // 16), max(2, H // 16), 12.0),
        (max(2, W // 8),  max(2, H // 8),  8.0),
        (max(2, W // 4),  max(2, H // 4),  5.0),
    ]

    for lw, lh, sigma in levels:
        small = rng.normal(0.0, sigma, size=(lh, lw)).astype(np.float32)
        small3 = small[:, :, None]
        up = _resize_bilinear_float(small3, W, H)[:, :, 0]
        tex += up

    tex += rng.normal(0.0, 6.0, size=(H, W)).astype(np.float32)

    tex -= tex.min()
    tex /= (tex.max() + 1e-6)
    tex = 0.85 + 0.30 * tex

    out = f * tex[..., None]
    return _to_u8(out)
