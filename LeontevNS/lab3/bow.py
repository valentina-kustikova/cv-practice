import cv2
import numpy as np
from sklearn.cluster import MiniBatchKMeans, KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import os

class BOW:
    def __init__(self, clusters_name="MiniBatch", clf_name="SVC", 
                 k_nearest=5, clusters=500, batch_size=1000,
                 descriptor_type="SIFT"):
        self.clusters = clusters
        self.batch_size = batch_size
        self.clusters_name = clusters_name
        self.clf_name = clf_name
        self.k_nearest = k_nearest
        self.descriptor_type = descriptor_type
        self.descriptors_list = None
        self.kmeans = None
        self.histograms = None
        self.clf = None
        self.scaler = None
        self.feature_extractor = None
        self.label_encoder = None
    
    def create_descriptor_extractor(self):
        if self.descriptor_type == "SIFT":
            return cv2.SIFT_create()
        elif self.descriptor_type == "ORB":
            return cv2.ORB_create(nfeatures=1000)
        elif self.descriptor_type == "SURF":
            try:
                return cv2.xfeatures2d.SURF_create()
            except:
                print("SURF недоступен, используется SIFT")
                return cv2.SIFT_create()
        else:
            print(f"Дескриптор {self.descriptor_type} не поддерживается, используется SIFT")
            return cv2.SIFT_create()
    
    def extract_descriptors(self, images, verbose=True):
        if self.feature_extractor is None:
            self.feature_extractor = self.create_descriptor_extractor()
        
        self.descriptors_list = []
        valid_indices = []
        
        iterator = tqdm(images, desc="Извлечение дескрипторов") if verbose else images
        
        for idx, img in enumerate(iterator):
            if len(img.shape) == 3:
                gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            else:
                gray = img
            
            keypoints, descriptors = self.feature_extractor.detectAndCompute(gray, None)
            
            if descriptors is not None and len(descriptors) > 10:
                self.descriptors_list.append(descriptors)
                valid_indices.append(idx)
            else:
                dummy_desc = np.zeros((10, 128)) if self.descriptor_type == "SIFT" else np.zeros((10, 32))
                self.descriptors_list.append(dummy_desc)
                valid_indices.append(idx)
        
        if verbose:
            total_descriptors = sum(len(desc) for desc in self.descriptors_list)
            print(f"Извлечено {total_descriptors} дескрипторов из {len(images)} изображений")
        
        return valid_indices
    
    def visualize_descriptors(self, img, max_keypoints=100):
        if self.feature_extractor is None:
            self.feature_extractor = self.create_descriptor_extractor()
        
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        else:
            gray = img
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        
        keypoints, descriptors = self.feature_extractor.detectAndCompute(gray, None)
        
        if descriptors is not None:
            if len(keypoints) > max_keypoints:
                keypoints = sorted(keypoints, key=lambda x: -x.response)[:max_keypoints]
            
            img_kp = cv2.drawKeypoints(
                img, keypoints, None,
                flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
                color=(0, 255, 0)
            )
            
            stats_text = f"Ключевых точек: {len(keypoints)}\nДескрипторов: {len(descriptors)}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            y0, dy = 30, 30
            for i, line in enumerate(stats_text.split('\n')):
                y = y0 + i * dy
                cv2.putText(img_kp, line, (10, y), font, 0.7, (0, 0, 255), 2)
            
            return img_kp, len(keypoints), len(descriptors)
        
        return img, 0, 0
    
    def bag_of_words(self, verbose=True):
        if not self.descriptors_list:
            raise ValueError("Сначала необходимо извлечь дескрипторы")
        
        all_descriptors = np.vstack(self.descriptors_list)
        
        if verbose:
            print(f"Всего дескрипторов для кластеризации: {len(all_descriptors)}")
            print(f"Размер дескриптора: {all_descriptors.shape[1]}")
        
        if self.clusters_name == "KMeans":
            self.kmeans = KMeans(
                n_clusters=self.clusters,
                random_state=42,
                n_init=10,
                max_iter=300,
                verbose=0
            )
        elif self.clusters_name == "MiniBatch":
            self.kmeans = MiniBatchKMeans(
                n_clusters=self.clusters,
                random_state=42,
                batch_size=self.batch_size,
                max_iter=100,
                verbose=0
            )
        else:
            raise ValueError(f"Метод кластеризации {self.clusters_name} не поддерживается")
        
        if verbose:
            print("Обучение K-Means...")
        
        self.kmeans.fit(all_descriptors)
        
        if verbose:
            print("Кластеризация завершена")
    
    def create_histograms(self, verbose=True):
        if self.kmeans is None:
            raise ValueError("Сначала необходимо создать словарь (bag_of_words)")
        
        self.histograms = []
        
        iterator = tqdm(self.descriptors_list, desc="Создание гистограмм") if verbose else self.descriptors_list
        
        for descriptors in iterator:
            if len(descriptors) > 0:
                words = self.kmeans.predict(descriptors)
                hist, _ = np.histogram(words, bins=np.arange(self.kmeans.n_clusters + 1))
            else:
                hist = np.zeros(self.kmeans.n_clusters)
            
            if hist.sum() > 0:
                hist = hist / hist.sum()
            
            self.histograms.append(hist)
        
        self.histograms = np.array(self.histograms)
        
        if verbose:
            print(f"Создано {len(self.histograms)} гистограмм")
            print(f"Размер гистограммы: {self.histograms.shape[1]}")
    
    def create_classifier(self):
        if self.clf_name == "SVC":
            self.clf = SVC(
                kernel='linear',
                probability=True,
                random_state=42,
                C=1.0
            )
        elif self.clf_name == "KNN":
            self.clf = KNeighborsClassifier(
                n_neighbors=self.k_nearest,
                weights='distance',
                metric='euclidean'
            )
        elif self.clf_name == "RandomForest":
            self.clf = RandomForestClassifier(
                n_estimators=100,
                random_state=42,
                max_depth=10
            )
        elif self.clf_name == "GradientBoosting":
            self.clf = GradientBoostingClassifier(
                n_estimators=100,
                random_state=42,
                learning_rate=0.1
            )
        else:
            raise ValueError(f"Классификатор {self.clf_name} не поддерживается")
    
    def train_bow_model(self, labels, verbose=True):
        if self.histograms is None:
            raise ValueError("Сначала необходимо создать гистограммы")
        
        X = self.histograms
        
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        self.create_classifier()
        
        if verbose:
            print(f"Обучение классификатора {self.clf_name}...")
        
        self.clf.fit(X_scaled, labels)
        
        if verbose:
            train_preds = self.clf.predict(X_scaled)
            train_acc = accuracy_score(labels, train_preds)
            print(f"Точность на обучающих данных: {train_acc:.4f}")
    
    def test_bow_model(self, images, labels=None, verbose=True):
        if self.clf is None or self.scaler is None or self.kmeans is None:
            raise ValueError("Модель не обучена. Сначала выполните train_bow_model")
        
        test_descriptors_list = []
        for img in images:
            if len(img.shape) == 3:
                gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            else:
                gray = img
            
            keypoints, descriptors = self.feature_extractor.detectAndCompute(gray, None)
            if descriptors is not None:
                test_descriptors_list.append(descriptors)
            else:
                test_descriptors_list.append(np.zeros((10, 128)))
        
        test_histograms = []
        for descriptors in test_descriptors_list:
            if len(descriptors) > 0:
                words = self.kmeans.predict(descriptors)
                hist, _ = np.histogram(words, bins=np.arange(self.kmeans.n_clusters + 1))
                if hist.sum() > 0:
                    hist = hist / hist.sum()
            else:
                hist = np.zeros(self.kmeans.n_clusters)
            test_histograms.append(hist)
        
        test_histograms = np.array(test_histograms)
        
        X_test_scaled = self.scaler.transform(test_histograms)
        predictions = self.clf.predict(X_test_scaled)
        
        if labels is not None:
            accuracy = accuracy_score(labels, predictions)
            
            if verbose:
                print(f"Точность на тестовых данных: {accuracy:.4f}")
                
                report = classification_report(labels, predictions, output_dict=False)
                print("\nОтчет о классификации:")
                print(report)
                
                cm = confusion_matrix(labels, predictions)
                print("\nConfusion Matrix:")
                print(cm)
            return predictions, accuracy
        
        return predictions
    
    def save_model(self, output_path):
        model_data = {
            'clf': self.clf,
            'scaler': self.scaler,
            'kmeans': self.kmeans,
            'feature_extractor_type': self.descriptor_type,
            'clusters': self.clusters,
            'clf_name': self.clf_name,
            'clusters_name': self.clusters_name
        }
        
        joblib.dump(model_data, output_path)
        print(f"Модель сохранена в {output_path}")
    
    def load_model(self, model_path):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Файл модели {model_path} не найден")
        
        model_data = joblib.load(model_path)
        
        self.clf = model_data['clf']
        self.scaler = model_data['scaler']
        self.kmeans = model_data['kmeans']
        self.descriptor_type = model_data.get('feature_extractor_type', 'SIFT')
        self.clusters = model_data.get('clusters', 500)
        self.clf_name = model_data.get('clf_name', 'SVC')
        self.clusters_name = model_data.get('clusters_name', 'MiniBatch')
        
        self.feature_extractor = self.create_descriptor_extractor()
        
        print(f"Модель загружена из {model_path}")
    
    def get_feature_importance(self):
        if self.clf_name == "RandomForest":
            importances = self.clf.feature_importances_
            return importances
        elif self.clf_name == "SVC" and hasattr(self.clf, 'coef_'):
            importances = np.abs(self.clf.coef_).mean(axis=0)
            return importances
        else:
            print(f"Важность признаков не доступна для классификатора {self.clf_name}")
            return None
    
    def plot_histograms(self, num_images=5, save_path=None):
        if self.histograms is None:
            print("Гистограммы не созданы")
            return
        
        fig, axes = plt.subplots(num_images, 1, figsize=(12, 3*num_images))
        
        if num_images == 1:
            axes = [axes]
        
        for i in range(min(num_images, len(self.histograms))):
            ax = axes[i]
            ax.bar(range(len(self.histograms[i])), self.histograms[i])
            ax.set_title(f"Гистограмма изображения {i+1}")
            ax.set_xlabel("Визуальное слово")
            ax.set_ylabel("Частота")
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()