import cv2
import os


def read_image(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Error: file '{file_path}' not found.")
    
    img = cv2.imread(file_path)

    if img is None: 
        raise ValueError(f"Error: file '{file_path}' damaged or not an image.")
    
    return img