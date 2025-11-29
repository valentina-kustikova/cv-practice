import os
import cv2
import numpy as np
from sklearn.preprocessing import LabelEncoder

class DataLoader:
    def __init__(self, data_path, train_split_file, test_split_file):
        self.data_path = data_path
        self.train_split_file = train_split_file
        self.test_split_file = test_split_file
        self.classes = ['kremlin', 'palace', 'cathedral']  # Fixed class names
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(self.classes)
    
    def load_split_files(self):
        with open(self.train_split_file, 'r') as f:
            train_files = [line.strip() for line in f.readlines()]
        
        with open(self.test_split_file, 'r') as f:
            test_files = [line.strip() for line in f.readlines()]
        
        return train_files, test_files
    
    def load_image(self, file_path):
        full_path = os.path.join(self.data_path, file_path)
        if not os.path.exists(full_path):
            print(f"Warning: File {full_path} not found")
            return None
        
        image = cv2.imread(full_path)
        if image is None:
            return None
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image
    
    def extract_label_from_path(self, file_path):
        filename = os.path.basename(file_path).lower()
        
        if 'kremlin' in filename:
            return 'kremlin'
        elif 'palace' in filename:
            return 'palace'
        elif 'cathedral' in filename:
            return 'cathedral'
        else:
            dir_name = os.path.dirname(file_path).lower()
            for cls in self.classes:
                if cls in dir_name:
                    return cls
        
        return None
    
    def load_data(self):
        train_files, test_files = self.load_split_files()
        
        train_images, train_labels = [], []
        test_images, test_labels = [], []
        
        print("Loading training data...")
        for file_path in train_files:
            image = self.load_image(file_path)
            label = self.extract_label_from_path(file_path)
            
            if image is not None and label is not None:
                train_images.append(image)
                train_labels.append(label)
        
        print("Loading test data...")
        for file_path in test_files:
            image = self.load_image(file_path)
            label = self.extract_label_from_path(file_path)
            
            if image is not None and label is not None:
                test_images.append(image)
                test_labels.append(label)
        
        train_labels_encoded = self.label_encoder.transform(train_labels)
        test_labels_encoded = self.label_encoder.transform(test_labels)
        
        print(f"Loaded {len(train_images)} training images")
        print(f"Loaded {len(test_images)} test images")
        
        return train_images, train_labels_encoded, test_images, test_labels_encoded
    
    def get_class_names(self):
        return self.classes