import numpy as np

from .base import ImageFilter


class InfraredFilm(ImageFilter):
    """Непроявленная плёнка: вуаль, подъём чёрного, цветовой сдвиг, зерно."""
    name = "film"

    def __init__(self, mix: float = 0.6, grain: float = 0.3,
                 seed: int = 42, veil: float = 0.25,
                 lift: int = 35, tint: str = "cyan"):
        self.mix = float(np.clip(mix, 0.0, 1.0))
        self.grain = float(max(0.0, grain))
        self.seed = int(seed)
        self.veil = float(np.clip(veil, 0.0, 1.0))
        self.lift = int(np.clip(lift, 0, 120))
        self.tint = tint

    def apply_filter(self, image: np.ndarray) -> np.ndarray:
        img = image.astype(np.float32)

        r, g, b = img[..., 0], img[..., 1], img[..., 2]
        
        # Создаем агрессивную маску ИК-отражения. 
        green_mask = np.clip(g * 2.0 - r * 0.5 - b * 0.5, 0, 255)
        
        # Базовая яркость кадра, где зеленый доминирует, а синий полностью уничтожен
        ir_base = (0.8 * g + 0.4 * r - 0.6 * b)
        
        # Объединяем базу и маску зелени
        ir_gray = ir_base + green_mask * 1.2
        ir_gray = np.clip(ir_gray, 0.0, 255.0)

        # Применяем жесткую S-кривую для контраста (переводим в диапазон 0-1 для удобства)
        norm_gray = ir_gray / 255.0

        contrast_gray = 1.0 / (1.0 + np.exp(-10.0 * (norm_gray - 0.45)))
        ir_gray = contrast_gray * 255.0

        # Собираем ЧБ, где mix управляет силой эффекта ИК над обычным ЧБ контрастом
        standard_gray = 0.299 * r + 0.587 * g + 0.114 * b
        base_mono = standard_gray * (1 - self.mix) + ir_gray * self.mix
        
        out = np.stack([base_mono, base_mono, base_mono], axis=-1)

        # 2) Подъём чёрного (эффект вуали в тенях)
        f_min = float(self.lift)
        out = f_min + out * (255.0 - f_min) / 255.0

        # 3) Эмульсионное ИК-свечение
        p = np.pad(out, ((1, 1), (1, 1), (0, 0)), mode="edge")
        blur = (p[:-2, 1:-1] + p[2:, 1:-1]
                + p[1:-1, :-2] + p[1:-1, 2:]
                + p[1:-1, 1:-1]) / 5.0
        out = out * 0.4 + blur * 0.6

        # 4) Цветовая вуаль непроявленной пленки
        if self.tint == "cyan":
            veil_color = np.array([130.0, 160.0, 170.0], dtype=np.float32)
        elif self.tint == "magenta":
            veil_color = np.array([170.0, 140.0, 160.0], dtype=np.float32)
        else:  # sepia
            veil_color = np.array([160.0, 145.0, 125.0], dtype=np.float32)

        v = self.veil
        out = out * (1 - v) + veil_color[None, None, :] * v

        # 5) Крупнозернистая структура эмульсии
        if self.grain > 0:
            rng = np.random.default_rng(self.seed)
            lum = base_mono / 255.0
            mask = np.clip(1.0 - 4.0 * (lum - 0.5) ** 2, 0.1, 1.0)
            noise = rng.normal(0.0, self.grain * 30.0, out.shape[:2]).astype(np.float32)
            out += noise[..., None] * mask[..., None]

        return self._u8(out)
