from abc import ABC, abstractmethod

import numpy as np


class ImageFilter(ABC):
    name = "base"

    @abstractmethod
    def apply_filter(self, image: np.ndarray) -> np.ndarray:
        ...

    @staticmethod
    def get_filter(name, **params):
        from .resize import Resize
        from .grayscale import RGB2GrayScale
        from .antique import Antique
        from .fade import FadeColor
        from .film import InfraredFilm
        from .matte import Matte
        from .scratches import Scratches
        from .neon import Neon

        reg = {
            "resize": Resize,
            "grayscale": RGB2GrayScale,
            "antique": Antique,
            "fade": FadeColor,
            "film": InfraredFilm,
            "matte": Matte,
            "scratches": Scratches,
            "neon": Neon,
        }
        if name not in reg:
            raise ValueError("unknown filter: " + name)
        return reg[name](**params)

    @staticmethod
    def _u8(x):
        return np.clip(x, 0, 255).astype(np.uint8)