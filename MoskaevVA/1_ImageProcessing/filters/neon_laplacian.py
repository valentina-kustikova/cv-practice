import numpy as np

from .base import ImageFilter


class Neon(ImageFilter):
    name = "neon"

    def __init__(self, threshold: float = 30.0, color=(0, 255, 255),
                 glow_radius: int = 2, glow_strength: float = 1.2,
                 halo_radius: int = 6, halo_strength: float = 0.8,
                 inner_glow: float = 0.6):
        self.threshold = float(threshold)
        self.color = np.array(color, dtype=np.float32)
        self.glow_radius = int(max(0, glow_radius))
        self.glow_strength = float(glow_strength)
        self.halo_radius = int(max(0, halo_radius))
        self.halo_strength = float(halo_strength)
        self.inner_glow = float(inner_glow)

    # ---------- Лапласиан (базовые сдвиги матриц) ----------
    def _laplacian(self, gray: np.ndarray) -> np.ndarray:
        k = np.array([[1,  1, 1],
                      [1, -8, 1],
                      [1,  1, 1]], dtype=np.float32)
        h, w = gray.shape
        p = np.pad(gray, 1, mode="edge")
        out = np.zeros_like(gray, dtype=np.float32)
        for dy in range(3):
            for dx in range(3):
                if k[dy, dx] != 0:
                    out += k[dy, dx] * p[dy:dy + h, dx:dx + w]
        return out

    # ---------- Мощный скользящий Box Blur (базовый NumPy) ----------
    def _box_blur(self, img: np.ndarray, r: int) -> np.ndarray:
        if r <= 0:
            return img.copy()
        
        # 1) Размытие по горизонтали (ось X)
        padded_x = np.pad(img, ((0, 0), (r, r)), mode='edge')
        cum_x = np.cumsum(padded_x, axis=1)
        blur_x = (cum_x[:, 2*r:] - cum_x[:, :-2*r]) / (2*r)
        
        # 2) Размытие по вертикали (ось Y)
        padded_y = np.pad(blur_x, ((r, r), (0, 0)), mode='edge')
        cum_y = np.cumsum(padded_y, axis=0)
        blur_y = (cum_y[2*r:, :] - cum_y[:-2*r, :]) / (2*r)
        
        return blur_y

    def apply_filter(self, image: np.ndarray) -> np.ndarray:
        img = image.astype(np.float32)
        
        # Перевод в ЧБ на чистых весах
        gray = (0.299 * img[..., 0] + 0.587 * img[..., 1] + 0.114 * img[..., 2])

        # Выделение контуров
        lap = self._laplacian(gray)
        edge_mask = np.clip((np.abs(lap) - self.threshold) * 3.0, 0, 255) / 255.0

        # Нормализуем цвет неона под диапазон [0.0, 1.0]
        c = self.color / 255.0

        # --- Слои неонового света ---
        # 1) Жесткое ядро контура
        neon_edge = edge_mask[..., None] * c

        # 2) Внутреннее мягкое свечение линий
        inner = self._box_blur(edge_mask, self.glow_radius * 2)
        neon_inner = inner[..., None] * c * self.glow_strength

        # 3) Огромное объемное внешнее гало (Halo)
        neon_halo = np.zeros_like(neon_edge)
        if self.halo_radius > 0:
            # Масштабируем радиус окна под разрешение картинки
            r_wide = self.halo_radius * 4
            wide = self._box_blur(edge_mask, r_wide)
            
            # Аккуратно вырезаем центр контура, формируя мягкое внешнее облако
            halo_mask = np.clip(wide - inner * 0.2, 0.0, 1.0)
            neon_halo = halo_mask[..., None] * c * self.halo_strength

        # --- БЛЕНДИНГ SCREEN (Экранное наложение без пересветов) ---
        neon_total = 1.0 - (1.0 - neon_edge) * (1.0 - neon_inner) * (1.0 - neon_halo)
        
        # Притемняем исходную фотографию мухомора до 15% под глубокую ночь
        bg = (img / 255.0) * 0.15
        
        # Накладываем светящийся неон поверх темного кадра
        out = (1.0 - (1.0 - bg) * (1.0 - neon_total)) * 255.0
        return self._u8(out)
