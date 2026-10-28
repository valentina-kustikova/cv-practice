from abc import ABC, abstractmethod


class ImageFilter(ABC):
    @abstractmethod
    def apply_filter(self, img, **kwargs):
        pass

    @staticmethod
    def get_filter(filter_name):
        from filters.color import Antique, FadeColor, Infrared, RGB2GrayScale
        from filters.effects import AgedPhoto, Matte, NeonEffect
        from filters.geometry import Resize

        filters = {
            "resize": Resize,
            "grayscale": RGB2GrayScale,
            "antique": Antique,
            "fade": FadeColor,
            "infrared": Infrared,
            "matte": Matte,
            "aged": AgedPhoto,
            "neon": NeonEffect,
        }
        if filter_name not in filters:
            raise ValueError(f"Unknown filter: {filter_name}")
        return filters[filter_name]()
