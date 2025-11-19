import cv2
import numpy as np
import os
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import LabelEncoder
from .base_classifier import BaseClassifier

class CNNClassifier(BaseClassifier):
    def __init__(self, model_dir='cnn_model', image_size=(224, 224),
                 learning_rate=0.001, dropout_rate=0.5):
        """
        Инициализация CNN классификатора на основе VGG16

        Args:
            model_dir (str): Директория для сохранения/загрузки моделей
            image_size (tuple): Размер входного изображения (ширина, высота)
            learning_rate (float): Скорость обучения
            dropout_rate (float): Коэффициент dropout
        """
        super().__init__(model_dir)
        self.image_size = image_size
        self.learning_rate = learning_rate
        self.dropout_rate = dropout_rate
        self.model = None
        self.label_encoder = LabelEncoder()

    def create_model(self, n_classes):
        """
        Создание модели на основе VGG16 с transfer learning

        Args:
            n_classes (int): Количество классов
        """
        # Загрузка базовой модели VGG16 без верхних слоев
        base_model = VGG16(weights='imagenet', include_top=False)

        # Заморозка весов базовой модели
        for layer in base_model.layers:
            layer.trainable = False

        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(512, activation='relu')(x)
        x = Dropout(self.dropout_rate)(x)
        predictions = Dense(n_classes, activation='softmax')(x)

        # Создание финальной модели
        self.model = Model(inputs=base_model.input, outputs=predictions)

        self.model.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

    def preprocess_image(self, image):
        """
        Предобработка изображения для CNN

        Args:
            image: Исходное изображение

        Returns:
            numpy.ndarray: Предобработанное изображение
        """
        # Изменение размера
        image = cv2.resize(image, self.image_size)

        # Конвертация BGR в RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Предобработка для VGG16
        image = preprocess_input(image)

        return image

    def train(self, train_paths, train_labels):
        """
        Обучение классификатора

        Args:
            train_paths (list): Пути к обучающим изображениям
            train_labels (list): Метки обучающих изображений
        """
        print("Начало обучения CNN классификатора...")

        self.class_names = sorted(list(set(train_labels)))
        encoded_labels = self.label_encoder.fit_transform(train_labels)

        X_train = np.array([
            self.preprocess_image(self.load_image(path))
            for path in train_paths
        ])

        y_train = np.eye(len(self.class_names))[encoded_labels]

        if self.model is None:
            self.create_model(len(self.class_names))

        self.model.fit(
            X_train, y_train,
            batch_size=16,
            epochs=10,
            validation_split=0.2
        )

        self.save_model()
        print("Обучение завершено")

    def test(self, test_paths, test_labels=None):
        """
        Тестирование классификатора

        Args:
            test_paths (list): Пути к тестовым изображениям
            test_labels (list, optional): Метки тестовых изображений

        Returns:
            tuple: Предсказанные метки и точность (если доступны истинные метки)
        """
        if self.model is None:
            self.load_model()

        # Подготовка тестовых данных
        X_test = np.array([
            self.preprocess_image(self.load_image(path))
            for path in test_paths
        ])

        # Получение предсказаний
        predictions_prob = self.model.predict(X_test)
        predictions_idx = np.argmax(predictions_prob, axis=1)
        predictions = self.label_encoder.inverse_transform(predictions_idx)

        accuracy = None
        if test_labels is not None:
            correct = sum(1 for p, t in zip(predictions, test_labels) if p == t)
            accuracy = correct / len(test_labels)
            print(f"\nТочность классификации: {accuracy:.3f}")
            print("\nОтчет по классификации:")
            print(self.evaluate(test_labels, predictions, self.class_names))

        return predictions, accuracy

    def save_model(self):
        """Сохранение модели в файл"""
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)

        model_path = os.path.join(self.model_dir, 'cnn_model.h5')
        self.model.save(model_path)

        metadata = {
            'class_names': self.class_names,
            'label_encoder': self.label_encoder,
            'image_size': self.image_size
        }
        np.save(os.path.join(self.model_dir, 'metadata.npy'), metadata)

    def load_model(self):
        """Загрузка модели из файла"""
        model_path = os.path.join(self.model_dir, 'cnn_model.h5')
        if not os.path.exists(model_path):
            abs_path = os.path.abspath(model_path)
            raise ValueError(f"Модель не найдена по пути: {abs_path}\nПроверьте, что файл cnn_model.h5 существует в директории: {os.path.abspath(self.model_dir)}")

        self.model = load_model(model_path)
        npy_path = os.path.join(self.model_dir, 'metadata.npy')
        json_path = os.path.join(self.model_dir, 'metadata.json')
        if os.path.exists(npy_path):
            metadata = np.load(npy_path, allow_pickle=True).item()
            self.class_names = metadata.get('class_names')
            self.label_encoder = metadata.get('label_encoder', LabelEncoder())
            self.image_size = tuple(metadata.get('image_size', self.image_size))
        elif os.path.exists(json_path):
            import json
            with open(json_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            self.class_names = metadata.get('class_names')
            self.label_encoder = LabelEncoder()
            if self.class_names is not None:
                self.label_encoder.fit(self.class_names)
            self.image_size = tuple(metadata.get('image_size', self.image_size))
        else:
            raise ValueError('Metadata для модели не найдена (нужен metadata.npy или metadata.json)')
