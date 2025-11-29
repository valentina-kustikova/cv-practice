from typing import List, Dict, Any

from abstract import ClassificationStrategy


# ============================================================================
# Заглушка для нейросетевой стратегии классификации.
# Определяет интерфейс NeuralNetworkStrategy, но методы не реализованы (TODO).
# Предусмотрено для будущего расширения с использованием глубоких нейронных сетей.
# Все методы выбрасывают NotImplementedError.
# ============================================================================

class NeuralNetworkStrategy(ClassificationStrategy):
    """Стратегия классификации на основе нейронной сети (заглушка)"""

    def __init__(self, model_architecture: str = 'resnet'):
        """
        Args:
            model_architecture: Архитектура нейросети ('resnet', 'vgg', etc.)
        """
        self.model_architecture = model_architecture
        self.model = None
        self.class_names = None

    def train(self, train_data: List[str], train_labels: List[str]) -> None:
        raise NotImplementedError("Нейросетевой подход пока не реализован")

    def predict(self, image_path: str) -> str:
        raise NotImplementedError("Нейросетевой подход пока не реализован")

    def save(self, filepath: str) -> None:
        raise NotImplementedError("Нейросетевой подход пока не реализован")

    def load(self, filepath: str) -> None:
        raise NotImplementedError("Нейросетевой подход пока не реализован")

    def get_params(self) -> Dict[str, Any]:
        return {
            'algorithm': 'neural',
            'architecture': self.model_architecture
        }
