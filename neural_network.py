import cv2
import numpy as np
import pickle
from pathlib import Path
from typing import List, Tuple, Optional
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.applications import ResNet50, VGG16, MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import os

# Ограничение использования памяти GPU и CPU
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'

# Ограничение использования памяти TensorFlow
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(f"Ошибка настройки GPU: {e}")

# Убираем ограничения для максимальной производительности
# tf.config.threading.set_inter_op_parallelism_threads(2)
# tf.config.threading.set_intra_op_parallelism_threads(2)


class NeuralNetworkClassifier:
    
    def __init__(self, 
                 base_model_name: str = 'ResNet50',
                 img_size: Tuple[int, int] = (224, 224),
                 num_classes: int = 3,
                 learning_rate: float = 0.0001):
        self.base_model_name = base_model_name
        self.img_size = img_size
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.model = None
        self.base_model = None
        self._build_model()
    
    def _build_model(self):
        if self.base_model_name == 'ResNet50':
            self.base_model = ResNet50(
                weights='imagenet',
                include_top=False,
                input_shape=(*self.img_size, 3)
            )
        elif self.base_model_name == 'VGG16':
            self.base_model = VGG16(
                weights='imagenet',
                include_top=False,
                input_shape=(*self.img_size, 3)
            )
        elif self.base_model_name == 'MobileNetV2':
            self.base_model = MobileNetV2(
                weights='imagenet',
                include_top=False,
                input_shape=(*self.img_size, 3)
            )
        else:
            raise ValueError(f"Неподдерживаемая модель: {self.base_model_name}")
        
        self.base_model.trainable = False
        
        inputs = keras.Input(shape=(*self.img_size, 3))
        x = self.base_model(inputs, training=False)
        
        # Используем GlobalMaxPooling2D для дополнительной информации
        x1 = layers.GlobalAveragePooling2D()(x)
        x2 = layers.GlobalMaxPooling2D()(x)
        x = layers.Concatenate()([x1, x2])
        
        # Улучшенная архитектура для максимальной точности
        x = layers.Dense(1024, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.5)(x)
        
        x = layers.Dense(512, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.4)(x)
        
        x = layers.Dense(256, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.3)(x)
        
        x = layers.Dense(128, activation='relu')(x)
        x = layers.Dropout(0.2)(x)
        
        outputs = layers.Dense(self.num_classes, activation='softmax')(x)
        
        self.model = keras.Model(inputs, outputs)
        
        # Оптимальный learning rate для transfer learning
        initial_lr = self.learning_rate
        self.model.compile(
            optimizer=Adam(learning_rate=initial_lr, beta_1=0.9, beta_2=0.999),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
    
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        # Простое изменение размера до нужного размера
        resized = cv2.resize(image, self.img_size, interpolation=cv2.INTER_AREA)
        
        # Конвертация в RGB
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        
        # Нормализация для предобученных моделей ImageNet
        # Mean и std для ImageNet
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        
        normalized = (rgb.astype(np.float32) / 255.0 - mean) / std
        
        return normalized
    
    def prepare_data(self, images: List[np.ndarray], labels: List[int]) -> Tuple[np.ndarray, np.ndarray]:
        X = np.array([self.preprocess_image(img) for img in images])
        y = np.array(labels)
        return X, y
    
    def train(self, 
              train_images: List[np.ndarray], 
              train_labels: List[int],
              validation_images: Optional[List[np.ndarray]] = None,
              validation_labels: Optional[List[int]] = None,
              epochs: int = 50,
              batch_size: int = 16,
              use_augmentation: bool = True,
              use_class_weights: bool = True,
              fine_tune_after: Optional[int] = None):
        print("Подготовка данных...")
        X_train, y_train = self.prepare_data(train_images, train_labels)
        
        validation_data = None
        if validation_images is not None and validation_labels is not None:
            X_val, y_val = self.prepare_data(validation_images, validation_labels)
            validation_data = (X_val, y_val)
        else:
            # Разделяем тренировочные данные на train и validation (75/25 для большего валидационного набора)
            from sklearn.model_selection import train_test_split
            X_train, X_val, y_train, y_val = train_test_split(
                X_train, y_train, test_size=0.25, random_state=42, stratify=y_train
            )
            validation_data = (X_val, y_val)
            print(f"Разделение данных: {len(X_train)} тренировочных, {len(X_val)} валидационных")
        
        # Вычисление весов классов для балансировки - сбалансированный подход
        class_weights = None
        if use_class_weights:
            unique_labels = np.unique(y_train)
            class_weights_dict = compute_class_weight(
                'balanced',
                classes=unique_labels,
                y=y_train
            )
            # Нормализуем веса для стабильного обучения
            max_weight = max(class_weights_dict)
            class_weights_dict = class_weights_dict / max_weight * 2.5  # Максимум 2.5
            class_weights = dict(zip(unique_labels, class_weights_dict))
            class_counts = np.bincount(y_train)
            print(f"Веса классов: {class_weights}")
            print(f"Количество примеров по классам: {dict(zip(unique_labels, class_counts))}")
        
        print(f"Обучение модели {self.base_model_name}...")
        print(f"Размер тренировочной выборки: {len(X_train)}")
        
        # Callbacks - оптимизированные для достижения 90% точности
        callbacks = [
            EarlyStopping(
                monitor='val_accuracy' if validation_data else 'accuracy',
                patience=100,  # Очень большое терпение
                restore_best_weights=True,
                verbose=1,
                mode='max',
                min_delta=0.001  # Минимальное улучшение 0.1%
            ),
            ReduceLROnPlateau(
                monitor='val_loss' if validation_data else 'loss',
                factor=0.3,  # Более агрессивное уменьшение LR
                patience=10,
                min_lr=1e-7,
                verbose=1,
                mode='min',
                min_delta=0.001
            ),
            ModelCheckpoint(
                'best_model.h5',
                monitor='val_accuracy' if validation_data else 'accuracy',
                save_best_only=True,
                mode='max',
                verbose=1
            )
        ]
        
        if use_augmentation:
            # Максимально агрессивная аугментация для увеличения разнообразия данных
            datagen = ImageDataGenerator(
                rotation_range=30,
                width_shift_range=0.3,
                height_shift_range=0.3,
                horizontal_flip=True,
                vertical_flip=True,
                zoom_range=0.3,
                brightness_range=[0.7, 1.3],
                shear_range=0.2,
                fill_mode='nearest',
                channel_shift_range=0.2
            )
            train_generator = datagen.flow(X_train, y_train, batch_size=batch_size)
            steps_per_epoch = max(1, len(X_train) // batch_size)
            
            history = self.model.fit(
                train_generator,
                steps_per_epoch=steps_per_epoch,
                epochs=epochs,
                validation_data=validation_data,
                class_weight=class_weights,
                callbacks=callbacks,
                verbose=1
            )
        else:
            history = self.model.fit(
                X_train, y_train,
                batch_size=batch_size,
                epochs=epochs,
                validation_data=validation_data,
                class_weight=class_weights,
                callbacks=callbacks,
                verbose=1
            )
        
        # Fine-tuning: размораживаем часть слоев базовой модели
        if fine_tune_after is not None and len(history.history['loss']) >= fine_tune_after:
            print(f"\n{'='*60}")
            print(f"Fine-tuning: размораживаем слои после {fine_tune_after} эпохи...")
            print(f"{'='*60}\n")
            
            # Размораживаем верхние слои базовой модели
            self.base_model.trainable = True
            # Замораживаем первые слои, размораживаем только верхние
            num_unfrozen = 30
            total_layers = len(self.base_model.layers)
            for i, layer in enumerate(self.base_model.layers):
                if i < total_layers - num_unfrozen:
                    layer.trainable = False
                else:
                    layer.trainable = True
            
            print(f"Разморожено {num_unfrozen} верхних слоев из {total_layers}")
            
            # Перекомпилируем с меньшим learning rate для fine-tuning
            fine_tune_lr = self.learning_rate * 0.1 * 0.5  # Еще меньше для fine-tuning
            self.model.compile(
                optimizer=Adam(learning_rate=fine_tune_lr, beta_1=0.9, beta_2=0.999),
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy']
            )
            
            print(f"Продолжаем обучение с fine-tuning (learning rate: {fine_tune_lr})...")
            # Продолжаем обучение с меньшим learning rate
            fine_tune_epochs = epochs - fine_tune_after
            fine_tune_history = self.model.fit(
                train_generator if use_augmentation else (X_train, y_train),
                steps_per_epoch=steps_per_epoch if use_augmentation else None,
                epochs=epochs,
                initial_epoch=fine_tune_after,
                validation_data=validation_data,
                class_weight=class_weights,
                callbacks=callbacks,
                verbose=1
            )
            
            # Объединяем историю
            for key in history.history.keys():
                history.history[key].extend(fine_tune_history.history[key])
        
        print("Обучение завершено!")
        return history
    
    def predict(self, images: List[np.ndarray]) -> np.ndarray:
        if self.model is None:
            raise ValueError("Модель не обучена. Сначала вызовите train()")
        
        X = np.array([self.preprocess_image(img) for img in images])
        predictions = self.model.predict(X, verbose=0)
        predicted_labels = np.argmax(predictions, axis=1)
        return predicted_labels
    
    def predict_proba(self, images: List[np.ndarray]) -> np.ndarray:
        if self.model is None:
            raise ValueError("Модель не обучена. Сначала вызовите train()")
        
        X = np.array([self.preprocess_image(img) for img in images])
        predictions = self.model.predict(X, verbose=0)
        return predictions
    
    def evaluate(self, images: List[np.ndarray], labels: List[int]) -> float:
        X, y = self.prepare_data(images, labels)
        predictions = self.model.predict(X, verbose=0)
        predicted_labels = np.argmax(predictions, axis=1)
        accuracy = accuracy_score(y, predicted_labels)
        return accuracy
    
    def save(self, filepath: str):
        # Сохраняем в новом формате Keras
        if filepath.endswith('.h5'):
            self.model.save(filepath)
        else:
            self.model.save(filepath, save_format='h5')
        print(f"Модель сохранена в {filepath}")
    
    def load(self, filepath: str):
        self.model = keras.models.load_model(filepath)
        print(f"Модель загружена из {filepath}")
