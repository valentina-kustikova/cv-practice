"""
Модуль для реализации нейросетевого классификатора на основе ResNet50.
"""

import numpy as np
import cv2
from typing import List, Tuple, Optional
import os
from tensorflow import keras
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


class NeuralNetworkClassifier:
    """Класс для классификации изображений с использованием нейронной сети ResNet50."""
    
    def __init__(self, num_classes: int = 3, img_size: Tuple[int, int] = (224, 224), 
                 freeze_base: bool = True):
        """
        Инициализация классификатора.
        
        Args:
            num_classes: Количество классов
            img_size: Размер входных изображений
            freeze_base: Заморозить ли базовые слои ResNet50
        """
        self.num_classes = num_classes
        self.img_size = img_size
        self.freeze_base = freeze_base
        self.model = None
        self.is_trained = False
        self.history = None
    
    def build_model(self):
        """Построение модели на основе ResNet50 с transfer learning."""
        print("Построение модели ResNet50...")
        
        # Загружаем предобученную ResNet50 без верхних слоев
        base_model = ResNet50(
            weights='imagenet',
            include_top=False,
            input_shape=(self.img_size[0], self.img_size[1], 3)
        )
        
        # Замораживаем базовые слои, если требуется
        if self.freeze_base:
            for layer in base_model.layers:
                layer.trainable = False
        else:
            # Размораживаем последние слои для тонкой настройки
            for layer in base_model.layers[-20:]:
                layer.trainable = True
        
        # Добавляем собственные слои для классификации
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(512, activation='relu')(x)
        x = Dropout(0.5)(x)
        x = Dense(256, activation='relu')(x)
        x = Dropout(0.3)(x)
        predictions = Dense(self.num_classes, activation='softmax')(x)
        
        # Создаем модель
        self.model = Model(inputs=base_model.input, outputs=predictions)
        
        # Компилируем модель
        self.model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        print("Модель построена успешно")
        self.model.summary()
    
    def preprocess_image(self, img_path: str) -> np.ndarray:
        """
        Предобработка изображения для ResNet50.
        
        Args:
            img_path: Путь к изображению
            
        Returns:
            Предобработанное изображение
        """
        img = cv2.imread(img_path)
        if img is None:
            raise ValueError(f"Не удалось загрузить изображение: {img_path}")
        
        # Конвертируем BGR в RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Изменяем размер
        img = cv2.resize(img, self.img_size)
        
        # Препроцессинг для ResNet50
        img = preprocess_input(img)
        
        return img
    
    def prepare_data(self, data_list: List[Tuple[str, int]], 
                    augment: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """
        Подготовка данных для обучения/тестирования.
        
        Args:
            data_list: Список кортежей (путь к изображению, класс)
            augment: Использовать ли аугментацию данных
            
        Returns:
            Кортеж (images, labels)
        """
        images = []
        labels = []
        
        for img_path, label in data_list:
            try:
                img = self.preprocess_image(img_path)
                images.append(img)
                labels.append(label)
            except Exception as e:
                print(f"Ошибка при загрузке {img_path}: {e}")
                continue
        
        return np.array(images), np.array(labels)
    
    def get_data_generator(self, data_list: List[Tuple[str, int]], 
                          batch_size: int = 16, augment: bool = False):
        """
        Создание генератора данных с аугментацией.
        
        Args:
            data_list: Список кортежей (путь к изображению, класс)
            batch_size: Размер батча
            augment: Использовать ли аугментацию
            
        Returns:
            Генератор данных
        """
        # Подготавливаем данные
        images, labels = self.prepare_data(data_list, augment=False)
        
        # Создаем генератор данных
        if augment:
            datagen = ImageDataGenerator(
                rotation_range=20,
                width_shift_range=0.2,
                height_shift_range=0.2,
                horizontal_flip=True,
                zoom_range=0.2,
                fill_mode='nearest'
            )
        else:
            datagen = ImageDataGenerator()
        
        generator = datagen.flow(images, labels, batch_size=batch_size, shuffle=augment)
        
        return generator
    
    def train(self, train_data: List[Tuple[str, int]], 
              validation_data: Optional[List[Tuple[str, int]]] = None,
              epochs: int = 20, batch_size: int = 16,
              model_save_dir: str = "models"):
        """
        Обучение модели.
        
        Args:
            train_data: Тренировочные данные
            validation_data: Валидационные данные (опционально)
            epochs: Количество эпох
            batch_size: Размер батча
            model_save_dir: Директория для сохранения модели
        """
        if self.model is None:
            self.build_model()
        
        print(f"Обучение модели на {len(train_data)} изображениях...")
        
        # Создаем директорию для сохранения моделей
        os.makedirs(model_save_dir, exist_ok=True)
        
        # Подготавливаем данные
        train_images, train_labels = self.prepare_data(train_data)
        
        # Создаем генераторы данных
        train_gen = self.get_data_generator(train_data, batch_size, augment=True)
        
        # Callbacks
        callbacks = [
            ModelCheckpoint(
                filepath=os.path.join(model_save_dir, 'best_resnet50_model.h5'),
                monitor='val_accuracy' if validation_data else 'accuracy',
                save_best_only=True,
                verbose=1
            ),
            EarlyStopping(
                monitor='val_accuracy' if validation_data else 'accuracy',
                patience=5,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss' if validation_data else 'loss',
                factor=0.5,
                patience=3,
                min_lr=0.00001,
                verbose=1
            )
        ]
        
        # Валидационные данные
        validation_gen = None
        validation_steps = None
        if validation_data:
            validation_images, validation_labels = self.prepare_data(validation_data)
            validation_gen = self.get_data_generator(validation_data, batch_size, augment=False)
            validation_steps = len(validation_data) // batch_size
        
        # Обучение
        steps_per_epoch = len(train_data) // batch_size
        
        self.history = self.model.fit(
            train_gen,
            steps_per_epoch=steps_per_epoch,
            epochs=epochs,
            validation_data=validation_gen,
            validation_steps=validation_steps,
            callbacks=callbacks,
            verbose=1
        )
        
        self.is_trained = True
        print("Обучение завершено")
    
    def predict(self, img_path: str) -> int:
        """
        Предсказание класса для изображения.
        
        Args:
            img_path: Путь к изображению
            
        Returns:
            Предсказанный класс
        """
        if not self.is_trained:
            raise ValueError("Модель не обучена. Сначала вызовите train()")
        
        img = self.preprocess_image(img_path)
        img = np.expand_dims(img, axis=0)
        
        predictions = self.model.predict(img, verbose=0)
        predicted_class = np.argmax(predictions[0])
        
        return int(predicted_class)
    
    def predict_batch(self, img_paths: List[str]) -> np.ndarray:
        """
        Предсказание классов для списка изображений.
        
        Args:
            img_paths: Список путей к изображениям
            
        Returns:
            Массив предсказанных классов
        """
        if not self.is_trained:
            raise ValueError("Модель не обучена. Сначала вызовите train()")
        
        images = []
        for img_path in img_paths:
            try:
                img = self.preprocess_image(img_path)
                images.append(img)
            except Exception as e:
                print(f"Ошибка при обработке {img_path}: {e}")
                images.append(np.zeros((self.img_size[0], self.img_size[1], 3)))
        
        images = np.array(images)
        predictions = self.model.predict(images, verbose=0)
        predicted_classes = np.argmax(predictions, axis=1)
        
        return predicted_classes
    
    def evaluate(self, test_data: List[Tuple[str, int]]) -> dict:
        """
        Оценка точности модели на тестовой выборке.
        
        Args:
            test_data: Тестовые данные
            
        Returns:
            Словарь с метриками
        """
        if not self.is_trained:
            raise ValueError("Модель не обучена. Сначала вызовите train()")
        
        print("Оценка качества на тестовой выборке...")
        
        img_paths = [path for path, _ in test_data]
        true_labels = np.array([label for _, label in test_data])
        
        predictions = self.predict_batch(img_paths)
        
        # Вычисляем метрики
        accuracy = accuracy_score(true_labels, predictions)
        precision = precision_score(true_labels, predictions, average='weighted', zero_division=0)
        recall = recall_score(true_labels, predictions, average='weighted', zero_division=0)
        f1 = f1_score(true_labels, predictions, average='weighted', zero_division=0)
        cm = confusion_matrix(true_labels, predictions)
        
        results = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'confusion_matrix': cm,
            'true_labels': true_labels,
            'predictions': predictions
        }
        
        return results
    
    def save(self, filepath: str):
        """Сохранение модели в файл."""
        if self.model is None:
            raise ValueError("Модель не построена")
        
        self.model.save(filepath)
        print(f"Модель сохранена в {filepath}")
    
    def load(self, filepath: str):
        """Загрузка модели из файла."""
        self.model = keras.models.load_model(filepath)
        self.is_trained = True
        print(f"Модель загружена из {filepath}")

