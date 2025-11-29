import cv2
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
from tqdm import tqdm
import pickle
import os

class BoWClassifier:
    def __init__(self, k=200):
        self.k = k
        self.sift = cv2.SIFT_create()
        self.kmeans = MiniBatchKMeans(n_clusters=k, batch_size=k*10, random_state=42)
        self.svm = SVC(kernel='rbf', C=100, gamma='scale', probability=True)
        self.vocab = None
        
    def extract_sift(self, images):
        descriptors_list = []
        all_keypoints = []
        
        for img in tqdm(images, desc="Extracting SIFT"):
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            kp, des = self.sift.detectAndCompute(gray, None)
            if des is not None:
                descriptors_list.append(des)
                all_keypoints.append(kp)
            else:
                descriptors_list.append(np.zeros((0, 128)))
                all_keypoints.append([])
        return descriptors_list, all_keypoints
    
    def build_vocabulary(self, descriptor_list):
        all_descriptors = np.vstack(descriptor_list)
        print(f"Создание словаря с помощью {len(all_descriptors)} дескрипторов...")
        self.kmeans.fit(all_descriptors)
        self.vocab = self.kmeans.cluster_centers_
    
    def create_histogram(self, descriptor_list):
        histograms = []
        for descriptors in tqdm(descriptor_list, desc="Создание гистограмм"):
            if descriptors.shape[0] == 0:
                hist = np.zeros(self.k)
            else:
                hist = np.zeros(self.k)
                words = self.kmeans.predict(descriptors)
                hist, _ = np.histogram(words, bins=self.k, range=(0, self.k), density=True)
            histograms.append(hist)
        return np.array(histograms)
    
    def train(self, train_images, train_labels, descriptor_list):
        print("Создание словаря")
        self.build_vocabulary(descriptor_list)
        
        print("Создание обучающих гистограмм")
        X_train = self.create_histogram(descriptor_list)
        self.svm.fit(X_train, train_labels)
        
    def predict(self, test_images):
        _, test_kp = self.extract_sift(test_images)
        test_desc_list = []
        for img in test_images:
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            _, des = self.sift.detectAndCompute(gray, None)
            test_desc_list.append(des if des is not None else np.zeros((0, 128)))
            
        X_test = self.create_histogram(test_desc_list)
        return self.svm.predict(X_test), self.svm.predict_proba(X_test)
    
    def save(self, path="bow_model.pkl"):
        with open(path, 'wb') as f:
            pickle.dump({
                'kmeans': self.kmeans,
                'svm': self.svm,
                'vocab': self.vocab
            }, f)
    
    def load(self, path="bow_model.pkl"):
        with open(path, 'rb') as f:
            data = pickle.load(f)
            self.kmeans = data['kmeans']
            self.svm = data['svm']
            self.vocab = data['vocab']