import numpy as np
import cv2

def bgr2grey(src_img):
    gray_img = np.dot(src_img[..., :3], [0.299, 0.587, 0.114]).astype(np.uint8)
    return np.stack((gray_img,)*3, axis=-1)

def resize(src_img, new_w, new_h):
    dst_img = np.zeros((new_h, new_w, 3), np.uint8)
    h, w = src_img.shape[:2]
    hRatio = h / new_h
    wRatio = w / new_w
    for i in range (new_h):
        for j in range (new_w):
            dst_img[i][j][0] = src_img[int(i * hRatio)][int(j * wRatio)][0]
            dst_img[i][j][1] = src_img[int(i * hRatio)][int(j * wRatio)][1]
            dst_img[i][j][2] = src_img[int(i * hRatio)][int(j * wRatio)][2]
    return dst_img

def apply_sepia(src_img):
    sepia_filter = np.array([[0.393, 0.349, 0.272],
                             [0.769, 0.686, 0.534],
                             [0.189, 0.168, 0.131]])
    sepia_img = np.dot(src_img[..., :3], sepia_filter)
    sepia_img[sepia_img > 255] = 255  # Обрезаем значения до 255
    return sepia_img.astype(np.uint8)

def apply_vignette(src_img):
    rows, cols = src_img.shape[:2]
    X_resultantMatrix = cv2.getGaussianKernel(cols, cols/4)
    Y_resultantMatrix = cv2.getGaussianKernel(rows, rows/4)
    resultantMatrix = Y_resultantMatrix * X_resultantMatrix.T
    uBound = np.full((rows, cols), 255)
    mask = 200 * resultantMatrix / np.linalg.norm(resultantMatrix)
    vignette_img = np.copy(src_img)

    for i in range(3):
        vignette_img[:, :, i] = np.minimum(vignette_img[:, :, i] * mask, uBound)
    
    return vignette_img.astype(np.uint8)

def pixelate_region(src_img, x, y, w, h, numOfPixels):
    region = src_img[y:y+h, x:x+w]
    temp = resize(region, numOfPixels, numOfPixels)
    pixelatedRegion = resize(temp, w, h)
    src_img[y:y+h, x:x+w] = pixelatedRegion
    return src_img