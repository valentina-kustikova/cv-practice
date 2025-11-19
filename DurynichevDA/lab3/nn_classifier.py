import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score, classification_report
import numpy as np
import os
from tqdm import tqdm

class NNClassifier:
    def __init__(self):
        self.model = None

    def build(self):
        base = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
        base.trainable = False
        x = base.output
        x = GlobalAveragePooling2D()(x)
        x = Dropout(0.5)(x)
        x = Dense(128, activation='relu')(x)
        outputs = Dense(3, activation='softmax')(x)
        self.model = Model(base.input, outputs)
        self.model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    def fit(self, train_data, val_data=None):
        print("\n" + "="*60)
        print("ОБУЧЕНИЕ НЕЙРОСЕТИ (EfficientNetB0)".center(60))
        print("="*60)
        self.build()

        def preprocess(path, label):
            img = tf.io.read_file(path)
            img = tf.image.decode_jpeg(img, channels=3)
            img = tf.image.resize(img, (224, 224))
            img = tf.keras.applications.efficientnet.preprocess_input(img)
            return img, label

        paths, labels = zip(*train_data)
        ds_train = tf.data.Dataset.from_tensor_slices((list(paths), list(labels)))
        ds_train = ds_train.map(preprocess).shuffle(1000).batch(32).prefetch(2)

        if val_data:
            paths_val, labels_val = zip(*val_data)
            ds_val = tf.data.Dataset.from_tensor_slices((list(paths_val), list(labels_val)))
            ds_val = ds_val.map(preprocess).batch(32)

        self.model.fit(ds_train, validation_data=ds_val if val_data else None,
                       epochs=20, callbacks=[EarlyStopping(patience=4, restore_best_weights=True)])
        self.model.save("models/nn_model.keras")
        print("Нейросеть обучена и сохранена!")

    def predict(self, test_data):
        print("\n" + "-"*60)
        print("ТЕСТИРОВАНИЕ НЕЙРОСЕТИ".center(60))
        print("-"*60)

        self.model = tf.keras.models.load_model("models/nn_model.keras")
        paths, labels = zip(*test_data)
        ds_test = tf.data.Dataset.from_tensor_slices(list(paths))
        ds_test = ds_test.map(lambda x: tf.keras.applications.efficientnet.preprocess_input(
            tf.image.resize(tf.image.decode_jpeg(tf.io.read_file(x), channels=3), (224, 224))
        )).batch(32)

        preds = np.argmax(self.model.predict(ds_test), axis=1)
        acc = accuracy_score(labels, preds)
        print(f"\nNN Accuracy: {acc:.4f}")
        print(classification_report(labels, preds,
              target_names=['Кремль', 'Архангельский собор', 'Дворец труда']))
        return acc