import urllib.request
import cv2
import os
from pathlib import Path
import numpy as np

from lib.settings import *

def resize_image_proportional(image, width=None, height=None, inter=cv2.INTER_AREA):
    (h, w) = image.shape[:2]

    if width is None and height is None:
        return image

    if width is None:
        ratio = height / float(h)
        dim = (int(w * ratio), height)
    else:
        ratio = width / float(w)
        dim = (width, int(h * ratio))

    resized = cv2.resize(image, dim, interpolation=inter)
    return resized


def load_images_from_folder(folder_path):
    """Выгружает все изображения из папки и возвращает их названия"""
    images = {}
    filenames = []

    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}

    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)

        if os.path.isfile(file_path):
            file_ext = Path(filename).suffix.lower()
            if file_ext in valid_extensions:
                try:
                    img = cv2.imread(file_path)
                    if img is not None:
                        images[filename] = img
                        filenames.append(filename)
                except Exception as e:
                    print(f"Ошибка при загрузке {filename}: {e}")

    print(f"Всего загружено изображений: {len(images)}")
    return filenames, images


def load_videos_from_folder(folder_path):
    """Выгружает все видео из папки и возвращает их названия"""
    videos = {}
    filenames = []

    valid_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv'}

    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)

        if os.path.isfile(file_path):
            file_ext = Path(filename).suffix.lower()
            if file_ext in valid_extensions:
                videos[filename] = file_path
                filenames.append(filename)

    print(f"Всего найдено видео: {len(videos)}")
    return filenames, videos


def uploading_models():
    """Скачивание моделей если их нет"""
    if not os.path.exists('models'):
        os.makedirs('models')

    for name, url in model_urls.items():
        model_path = f'models/{name}.onnx'
        if not os.path.exists(model_path):
            print(f"Скачивание {name}...")
            try:
                urllib.request.urlretrieve(url, model_path)
                print(f"Модель {name} скачана!")
            except Exception as e:
                print(f"Ошибка скачивания {name}: {e}")


def letterbox(srcimg, target_size=(640, 640), model_type='yolox'):
    """Универсальная функция letterbox для всех моделей"""
    if model_type == 'nanodet':
        img = srcimg.copy()
        top, left, newh, neww = 0, 0, target_size[0], target_size[1]
        if img.shape[0] != img.shape[1]:
            hw_scale = img.shape[0] / img.shape[1]
            if hw_scale > 1:
                newh, neww = target_size[0], int(target_size[1] / hw_scale)
                img = cv2.resize(img, (neww, newh), interpolation=cv2.INTER_AREA)
                left = int((target_size[1] - neww) * 0.5)
                img = cv2.copyMakeBorder(img, 0, 0, left, target_size[1] - neww - left, cv2.BORDER_CONSTANT, value=0)
            else:
                newh, neww = int(target_size[0] * hw_scale), target_size[1]
                img = cv2.resize(img, (neww, newh), interpolation=cv2.INTER_AREA)
                top = int((target_size[0] - newh) * 0.5)
                img = cv2.copyMakeBorder(img, top, target_size[0] - newh - top, 0, 0, cv2.BORDER_CONSTANT, value=0)
        else:
            img = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)
        letterbox_scale = [top, left, newh, neww]
        return img, letterbox_scale
    elif model_type == 'ssd':
        h, w = srcimg.shape[:2]
        scale_x = w / target_size[0]
        scale_y = h / target_size[1]
        return srcimg, (scale_x, scale_y)
    else:
        padded_img = np.ones((target_size[0], target_size[1], 3)).astype(np.float32) * 114.0
        ratio = min(target_size[0] / srcimg.shape[0], target_size[1] / srcimg.shape[1])
        resized_img = cv2.resize(
            srcimg, (int(srcimg.shape[1] * ratio), int(srcimg.shape[0] * ratio)), interpolation=cv2.INTER_LINEAR
        ).astype(np.float32)
        padded_img[: int(srcimg.shape[0] * ratio), : int(srcimg.shape[1] * ratio)] = resized_img
        return padded_img, ratio


def unletterbox(bbox, original_image_shape, letterbox_scale, model_type='yolox'):
    """Универсальная функция unletterbox для всех моделей"""
    if model_type == 'nanodet':
        ret = bbox.copy()
        h, w = original_image_shape
        top, left, newh, neww = letterbox_scale

        if h == w:
            ratio = h / newh
            ret = ret * ratio
            return ret.astype(np.int32)

        ratioh, ratiow = h / newh, w / neww
        ret[0] = max((ret[0] - left) * ratiow, 0)
        ret[1] = max((ret[1] - top) * ratioh, 0)
        ret[2] = min((ret[2] - left) * ratiow, w)
        ret[3] = min((ret[3] - top) * ratioh, h)
        return ret.astype(np.int32)
    elif model_type == 'ssd':
        ret = bbox.copy()
        scale_x, scale_y = letterbox_scale
        ret[0] = ret[0] * scale_x  # x1
        ret[1] = ret[1] * scale_y  # y1
        ret[2] = ret[2] * scale_x  # x2
        ret[3] = ret[3] * scale_y  # y2
        return ret.astype(np.int32)
    else:
        return (bbox / letterbox_scale).astype(np.int32)
