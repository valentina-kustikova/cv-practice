import os
import cv2
import numpy as np
from sklearn.cluster import KMeans
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

# 1. Функция для загрузки изображений
def load_images_from_folder(folder, label_prefix, label_value, limit=100):
    images = []
    labels = []
    count = 0
    for filename in os.listdir(folder):
        if filename.startswith(label_prefix) and count < limit:
            img_path = os.path.join(folder, filename)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                images.append(img)
                labels.append(label_value)
                count += 1
    return images, labels

# 2. Загрузка данных
train_folder = "/home/danila/Study/cv-practice/VyunovDV/lab3/data/train"
test_folder = "/home/danila/Study/cv-practice/VyunovDV/lab3/data/test"

# Загрузка тренировочных данных (200 изображений: 100 cats + 100 dogs)
train_cats, train_cat_labels = load_images_from_folder(train_folder, "cat", 0, 100)
train_dogs, train_dog_labels = load_images_from_folder(train_folder, "dog", 1, 100)
X_train = train_cats + train_dogs
y_train = train_cat_labels + train_dog_labels

# Загрузка тестовых данных (200 изображений: 100 cats + 100 dogs)
test_cats, test_cat_labels = load_images_from_folder(test_folder, "cat", 0, 100)
test_dogs, test_dog_labels = load_images_from_folder(test_folder, "dog", 1, 100)
X_test = test_cats + test_dogs
y_test = test_cat_labels + test_dog_labels

# 3. Извлечение признаков с помощью SIFT
def extract_descriptors(images):
    sift = cv2.SIFT_create()
    descriptors = []
    for img in images:
        _, des = sift.detectAndCompute(img, None)
        if des is not None:
            descriptors.extend(des)
    return np.array(descriptors)

# Извлечение дескрипторов из тренировочных изображений
print("Extracting descriptors...")
train_descriptors = extract_descriptors(X_train)

# 4. Создание словаря визуальных слов
print("Creating visual words...")
n_clusters = 40
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
print(train_descriptors.shape)
kmeans.fit(train_descriptors)

# 5. Создание гистограмм
def create_histogram(images, kmeans_model):
    sift = cv2.SIFT_create()
    histograms = []
    for img in images:
        _, des = sift.detectAndCompute(img, None)
        if des is not None:
            words = kmeans_model.predict(des)
            hist, _ = np.histogram(words, bins=range(n_clusters+1))
            histograms.append(hist)
        else:
            histograms.append(np.zeros(n_clusters))
    return np.array(histograms)

print("Creating histograms...")
X_train_hist = create_histogram(X_train, kmeans)
X_test_hist = create_histogram(X_test, kmeans)

# 6. Обучение классификатора
print("Training classifier...")
clf = SVC(kernel='linear', probability=True, random_state=42)
clf.fit(X_train_hist, y_train)

# 7. Оценка модели
print("Evaluating model...")
y_pred = clf.predict(X_test_hist)

print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:")
print(classification_report(y_test, y_pred))

# 8. Визуализация результатов
def visualize_example(image, keypoints, histogram):
    plt.figure(figsize=(12, 4))
    
    # Изображение с ключевыми точками
    plt.subplot(131)
    img_kp = cv2.drawKeypoints(image, keypoints, None)
    plt.imshow(img_kp, cmap='gray')
    plt.title("Keypoints")
    
    # Гистограмма
    plt.subplot(132)
    plt.bar(range(n_clusters), histogram)
    plt.title("Visual Words Histogram")
    
    # Исходное изображение
    plt.subplot(133)
    plt.imshow(image, cmap='gray')
    plt.title("Original Image")
    
    plt.tight_layout()
    plt.show()

# Пример визуализации для первого тестового изображения
sift = cv2.SIFT_create()
img = X_test[0]
kp, des = sift.detectAndCompute(img, None)
if des is not None:
    words = kmeans.predict(des)
    hist, _ = np.histogram(words, bins=range(n_clusters+1))
    visualize_example(img, kp, hist)
else:
    print("No keypoints found for visualization example.")