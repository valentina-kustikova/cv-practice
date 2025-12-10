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
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


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
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dropout(0.2)(x)
        x = layers.Dense(128, activation='relu')(x)
        x = layers.Dropout(0.2)(x)
        outputs = layers.Dense(self.num_classes, activation='softmax')(x)
        
        self.model = keras.Model(inputs, outputs)
        
        self.model.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
    
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        resized = cv2.resize(image, self.img_size)
        
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        
        normalized = rgb.astype(np.float32) / 255.0
        
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
              epochs: int = 20,
              batch_size: int = 16,
              use_augmentation: bool = True):
        print("Подготовка данных...")
        X_train, y_train = self.prepare_data(train_images, train_labels)
        
        validation_data = None
        if validation_images is not None and validation_labels is not None:
            X_val, y_val = self.prepare_data(validation_images, validation_labels)
            validation_data = (X_val, y_val)
        
        print(f"Обучение модели {self.base_model_name}...")
        print(f"Размер тренировочной выборки: {len(X_train)}")
        
        if use_augmentation:
            datagen = ImageDataGenerator(
                rotation_range=20,
                width_shift_range=0.2,
                height_shift_range=0.2,
                horizontal_flip=True,
                zoom_range=0.2,
                fill_mode='nearest'
            )
            train_generator = datagen.flow(X_train, y_train, batch_size=batch_size)
            steps_per_epoch = max(1, len(X_train) // batch_size)
            
            history = self.model.fit(
                train_generator,
                steps_per_epoch=steps_per_epoch,
                epochs=epochs,
                validation_data=validation_data,
                verbose=1
            )
        else:
            history = self.model.fit(
                X_train, y_train,
                batch_size=batch_size,
                epochs=epochs,
                validation_data=validation_data,
                verbose=1
            )
        
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
        self.model.save(filepath)
        print(f"Модель сохранена в {filepath}")
    
    def load(self, filepath: str):
        self.model = keras.models.load_model(filepath)
        print(f"Модель загружена из {filepath}")
