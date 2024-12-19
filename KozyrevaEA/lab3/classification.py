import cv2
import numpy as np
import os
from sklearn.cluster import KMeans
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


TRAIN_DIR = 'dogs-vs-cats/train'
IMG_SIZE = 128  
SIFT_FEATURES = 500  
NUM_CLUSTERS = 50  

def load_data():
    """Загрузка данных и формирование меток."""
    images, labels = [], []
    for filename in os.listdir(TRAIN_DIR):
        label = 0 if 'cat' in filename else 1
        img_path = os.path.join(TRAIN_DIR, filename)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        images.append(img)
        labels.append(label)
    return np.array(images), np.array(labels)


def extract_sift_features(images):
    """Извлечение SIFT признаков для каждого изображения."""
    sift = cv2.SIFT_create(nfeatures=SIFT_FEATURES)
    all_descriptors = []
    for img in images:
        keypoints, descriptors = sift.detectAndCompute(img, None)
        if descriptors is not None:
            all_descriptors.append(descriptors)
    return all_descriptors


def create_bow(descriptors_list):
    """Создание мешка слов."""
    all_descriptors = np.vstack(descriptors_list)
    kmeans = KMeans(n_clusters=NUM_CLUSTERS, random_state=42)
    kmeans.fit(all_descriptors)
    return kmeans


def get_bow_features(images, kmeans):
    """Создание гистограмм мешка слов для каждого изображения."""
    sift = cv2.SIFT_create(nfeatures=SIFT_FEATURES)
    features = []
    for img in images:
        _, descriptors = sift.detectAndCompute(img, None)
        if descriptors is not None:
            histogram = np.zeros(NUM_CLUSTERS)
            cluster_indices = kmeans.predict(descriptors)
            for idx in cluster_indices:
                histogram[idx] += 1
            features.append(histogram)
        else:
            features.append(np.zeros(NUM_CLUSTERS))
    return np.array(features)


def visualize_confusion_matrix(cm, classes):
    """Визуализация матрицы ошибок."""
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

if __name__ == "__main__":
    # 1. Загрузка данных
    images, labels = load_data()

    # Разделение на обучающую и тестовую выборки
    train_imgs, test_imgs, train_labels, test_labels = train_test_split(
        images, labels, test_size=0.5, random_state=42
    )

    # 2. Извлечение SIFT признаков
    train_descriptors = extract_sift_features(train_imgs)
    test_descriptors = extract_sift_features(test_imgs)

    # 3. Построение мешка слов
    kmeans = create_bow(train_descriptors)

    # 4. Преобразование в гистограммы
    train_features = get_bow_features(train_imgs, kmeans)
    test_features = get_bow_features(test_imgs, kmeans)

    # 5. Классификация SVM
    classifier = SVC(kernel='linear', random_state=42)
    classifier.fit(train_features, train_labels)
    predictions = classifier.predict(test_features)

    # 6. Оценка качества
    accuracy = accuracy_score(test_labels, predictions)
    print(f"Accuracy: {accuracy:.2f}")
    print(classification_report(test_labels, predictions, target_names=["Cat", "Dog"]))

    # 7. Визуализация
    cm = confusion_matrix(test_labels, predictions)
    visualize_confusion_matrix(cm, classes=["Cat", "Dog"])
