from abc import ABC, abstractmethod


class ImageFilter(ABC):
    name = ""

    @abstractmethod
    def apply_filter(self, image):
        pass

    @staticmethod
    def names():
        result = []
        for filter_class in ImageFilter.__subclasses__():
            result.append(filter_class.name)
        return result

    @staticmethod
    def get_filter(name, **params):
        for filter_class in ImageFilter.__subclasses__():
            if filter_class.name == name:
                try:
                    return filter_class(**params)
                except TypeError as error:
                    raise ValueError(f"неверные параметры: {error}")
        raise ValueError(f"неизвестный фильтр {name}, есть: {', '.join(ImageFilter.names())}")
