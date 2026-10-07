from abc import ABC, abstractmethod

class ImageFilter(ABC):
    """Абстрактный класс фильтра"""
    @staticmethod
    def get_filter(filter_name, **kwargs):
        from filters import Rescale, Grayscale, Antique, Fade, IR, Matte, Age, Neon
        if filter_name == "rescale":
            return Rescale(**kwargs)
        if filter_name == "grayscale":
            return Grayscale(**kwargs)
        if filter_name == "antique":
            return Antique(**kwargs)
        if filter_name == "fade":
            return Fade(**kwargs)
        if filter_name == "ir":
            return IR(**kwargs)
        if filter_name == "matte":
            return Matte(**kwargs)
        if filter_name == "age":
            return Age(**kwargs)
        if filter_name == "neon":
            return Neon(**kwargs)
        
    @abstractmethod
    def apply_filter(self, image):
        pass