import tensorflow as tf
from tensorflow.data import Dataset
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
import seaborn as sns
import os
import cv2

class NN:
    def __init__(self, model_name='MobileNetV2', input_shape=(224, 224, 3), num_classes=None):
        self.model_name = model_name
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = None
        self.history = None
        
        if model_name == 'MobileNetV2':
            base_model = MobileNetV2(
                weights='imagenet',
                include_top=False,
                input_shape=input_shape,
                pooling=None
            )
            
            base_model.trainable = False
            
            inputs = tf.keras.Input(shape=input_shape)
            
            x = base_model(inputs, training=False)
            
            x = GlobalAveragePooling2D()(x)
            x = Dropout(0.5)(x)
            x = Dense(256, activation='relu')(x)
            x = Dropout(0.3)(x)
            x = Dense(128, activation='relu')(x)
            
            if num_classes:
                outputs = Dense(num_classes, activation='softmax')(x)
            else:
                outputs = Dense(3, activation='softmax')(x)
            
            self.model = Model(inputs, outputs)
            
        elif model_name == 'CustomCNN':
            inputs = tf.keras.Input(shape=input_shape)
            
            # Блок 1
            x = tf.keras.layers.Conv2D(32, (3, 3), activation='relu')(inputs)
            x = tf.keras.layers.MaxPooling2D(2, 2)(x)
            
            # Блок 2
            x = tf.keras.layers.Conv2D(64, (3, 3), activation='relu')(x)
            x = tf.keras.layers.MaxPooling2D(2, 2)(x)
            
            # Блок 3
            x = tf.keras.layers.Conv2D(128, (3, 3), activation='relu')(x)
            x = tf.keras.layers.MaxPooling2D(2, 2)(x)
            
            # Полносвязные слои
            x = tf.keras.layers.Flatten()(x)
            x = tf.keras.layers.Dense(256, activation='relu')(x)
            x = tf.keras.layers.Dropout(0.5)(x)
            x = tf.keras.layers.Dense(128, activation='relu')(x)
            
            # Выходной слой
            if num_classes:
                outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
            else:
                outputs = tf.keras.layers.Dense(3, activation='softmax')(x)
            
            self.model = Model(inputs, outputs)
        else:
            raise ValueError(f"Модель {model_name} не поддерживается")
    
    def compile_model(self, learning_rate=0.0001):
        self.model.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
    
    def create_callbacks(self, checkpoint_path='best_model.keras', patience=5):
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=patience,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=patience // 2,
                min_lr=1e-7,
                verbose=1
            ),
            ModelCheckpoint(
                filepath=checkpoint_path,
                monitor='val_accuracy',
                save_best_only=True,
                mode='max',
                verbose=1
            )
        ]
        return callbacks
    
    def preprocess_images(self, images):
        processed_images = []
        
        for img in images:
            if len(img.shape) == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            elif img.shape[2] == 1:
                img = cv2.merge([img[:,:,0], img[:,:,0], img[:,:,0]])
            elif img.shape[2] == 4:
                img = img[:, :, :3]
            
            if img.shape[:2] != self.input_shape[:2]:
                img = cv2.resize(img, self.input_shape[:2])
            
            img = img.astype(np.float32)
            
            img = img / 255.0
            
            processed_images.append(img)
        
        return np.array(processed_images)
    
    def train_nn(self, train_images, train_labels, 
                 validation_data=None, epochs=20, batch_size=16):
        self.num_classes = len(np.unique(train_labels))
        print(f"Обнаружено {self.num_classes} классов")
        
        if self.model is None:
            self.init(self.model_name, self.input_shape, self.num_classes)
        
        print("Предобработка обучающих изображений...")
        X_train = self.preprocess_images(train_images)
        
        y_train = to_categorical(train_labels, self.num_classes)
        
        train_dataset = Dataset.from_tensor_slices((X_train, y_train))
        train_dataset = train_dataset.shuffle(buffer_size=len(X_train))
        train_dataset = train_dataset.batch(batch_size)
        train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)
        
        val_dataset = None
        if validation_data:
            val_images, val_labels = validation_data
            X_val = self.preprocess_images(val_images)
            y_val = to_categorical(val_labels, self.num_classes)
            val_dataset = Dataset.from_tensor_slices((X_val, y_val))
            val_dataset = val_dataset.batch(batch_size)
            val_dataset = val_dataset.prefetch(tf.data.AUTOTUNE)
        
        self.compile_model()
        
        callbacks = self.create_callbacks()
        
        print("Начало обучения...")
        self.history = self.model.fit(
            train_dataset,
            epochs=epochs,
            validation_data=val_dataset,
            callbacks=callbacks,
            verbose=1
        )
        
        if self.model_name == 'MobileNetV2' and val_dataset:
            self.fine_tune(train_dataset, val_dataset, epochs=10)
        
        print("Обучение завершено")
    
    def fine_tune(self, train_dataset, val_dataset, epochs=10):
        """Fine-tuning последних слоев модели"""
        print("Fine-tuning последних слоев...")
        
        for layer in self.model.layers[-20:]:
            layer.trainable = True
        
        self.model.compile(
            optimizer=Adam(learning_rate=1e-5),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        self.history_fine = self.model.fit(
            train_dataset,
            epochs=epochs,
            validation_data=val_dataset,
            callbacks=self.create_callbacks('best_model_fine.keras'),
            verbose=1
        )
    
    def save(self, model_path):
        self.model.save(model_path)
        print(f"Модель сохранена в {model_path}")
    
    def load(self, model_path):
        self.model = load_model(model_path)
        print(f"Модель загружена из {model_path}")
    
    def predict(self, images, batch_size=16):
        X = self.preprocess_images(images)
        predictions = self.model.predict(X, batch_size=batch_size, verbose=0)
        return np.argmax(predictions, axis=1), predictions
    
    def test_nn(self, test_images, test_labels=None, batch_size=16):
        predictions, probs = self.predict(test_images, batch_size)
        
        if test_labels is not None:
            accuracy = accuracy_score(test_labels, predictions)
            
            report = classification_report(test_labels, predictions, output_dict=True)
            
            cm = confusion_matrix(test_labels, predictions)
            
            return accuracy, predictions, report, cm
        
        return None, predictions, None, None
    
    def plot_training_history(self, save_path=None):
        if self.history is None:
            print("Нет истории обучения для визуализации")
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        axes[0].plot(self.history.history['accuracy'], label='Train Accuracy')
        if 'val_accuracy' in self.history.history:
            axes[0].plot(self.history.history['val_accuracy'], label='Val Accuracy')
        axes[0].set_title('Accuracy')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Accuracy')
        axes[0].legend()
        axes[0].grid(True)
        
        axes[1].plot(self.history.history['loss'], label='Train Loss')
        if 'val_loss' in self.history.history:
            axes[1].plot(self.history.history['val_loss'], label='Val Loss')
        axes[1].set_title('Loss')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Loss')
        axes[1].legend()
        axes[1].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"График сохранен в {save_path}")
        
        plt.show()
    
    def plot_confusion_matrix(self, cm, class_names, save_path=None):
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=class_names,
                   yticklabels=class_names)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Confusion matrix сохранена в {save_path}")
        
        plt.show()
    
    def model_summary(self):
        if self.model:
            self.model.summary()