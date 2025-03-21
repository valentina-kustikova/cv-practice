import os
import sys
import re
import cv2
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report

def extract_arguments():
    parser = argparse.ArgumentParser(description="Параметры для тренировки и тестирования модели")
    parser.add_argument("-tr", "--train_dir", type=str, default="train", help="Директория с обучающими изображениями")
    parser.add_argument("-te", "--test_dir", type=str, default="test", help="Директория с тестовыми изображениями")
    parser.add_argument("-c", "--num_clusters", type=int, default=41, help="Количество кластеров для KMeans")
    return parser.parse_args()

def fetch_images(dir_path):
    imgs = []
    for file in os.listdir(dir_path):
        full_path = os.path.join(dir_path, file)
        img = cv2.imread(full_path)
        if img is not None:
            imgs.append(cv2.resize(img, (256, 256)))
    return imgs

def fetch_train_labels(dir_path):
    labels = []
    for file in os.listdir(dir_path):
        base, _ = os.path.splitext(file)
        if base.startswith("cat"):
            labels.append(0)
        elif base.startswith("dog"):
            labels.append(1)
    return np.array(labels)

def fetch_test_labels(dir_path):
    labels = []
    for file in os.listdir(dir_path):
        base, _ = os.path.splitext(file)
        if re.match(r'^\d+$', base):
            num = int(base)
            labels.append(0 if num < 101 else 1)
    return np.array(labels)

def extract_sift_features(images):
    sift_extractor = cv2.SIFT_create()
    all_desc = []
    for img in images:
        keypoints, desc = sift_extractor.detectAndCompute(img, None)
        if desc is not None:
            all_desc.append(desc)
    return all_desc

def build_visual_dictionary(descriptors_list, clusters):
    concatenated = np.vstack(descriptors_list)
    kmeans_model = KMeans(n_clusters=clusters, random_state=42)
    return kmeans_model.fit(concatenated)

def compute_feature_histograms(descriptors_list, kmeans_model, clusters):
    histograms = []
    for desc in descriptors_list:
        hist = np.zeros(clusters)
        cluster_ids = kmeans_model.predict(desc)
        for cluster in cluster_ids:
            hist[cluster] += 1
        histograms.append(hist)
    return np.array(histograms)

def run():
    args = extract_arguments()
    
    # Загрузка изображений и соответствующих меток
    train_imgs = fetch_images(args.train_dir)
    test_imgs = fetch_images(args.test_dir)
    train_labels = fetch_train_labels(args.train_dir)
    test_labels = fetch_test_labels(args.test_dir)
    
    # Извлечение дескрипторов SIFT
    train_descriptors = extract_sift_features(train_imgs)
    test_descriptors = extract_sift_features(test_imgs)
    
    # Создание визуального словаря (мешка слов)
    dict_model = build_visual_dictionary(train_descriptors, args.num_clusters)
    
    # Построение гистограмм признаков
    train_features = compute_feature_histograms(train_descriptors, dict_model, args.num_clusters)
    test_features = compute_feature_histograms(test_descriptors, dict_model, args.num_clusters)
    
    # Обучение и предсказание с помощью SVC
    svc_classifier = SVC(kernel="rbf", random_state=42)
    svc_classifier.fit(train_features, train_labels)
    predictions = svc_classifier.predict(test_features)
    
    # Вывод классификационного отчёта и матрицы ошибок
    print(classification_report(test_labels, predictions, target_names=["Cat", "Dog"]))
    cm_array = confusion_matrix(test_labels, predictions)
    disp = ConfusionMatrixDisplay(cm_array, display_labels=["Cat", "Dog"])
    disp.plot(cmap="Reds")
    # Если backend не интерактивный, сохраняем график, иначе отображаем окно
    if matplotlib.get_backend().lower() == "agg":
        print("Сохраняем график в 'confusion_matrix.png'")
        plt.savefig("confusion_matrix.png")
    else:
        plt.show()

if __name__ == "__main__":
    sys.exit(run())