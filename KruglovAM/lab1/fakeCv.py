import numpy as np
import cv2 as cv

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
    sepia_filter = np.array([[0.272, 0.349, 0.393],
                             [0.534, 0.686, 0.769],
                             [0.131, 0.168, 0.189]])
    sepia_img = np.dot(src_img[..., :3], sepia_filter)
    sepia_img[sepia_img > 255] = 255  # Обрезаем значения до 255
    return sepia_img.astype(np.uint8)

def apply_vignette(src_img, radius):
    rows, cols = src_img.shape[:2]
    vignette_img = np.copy(src_img)

    height = src_img.shape[0]
    width = src_img.shape[1]
    center_x = width // 2
    center_y = height // 2
    radius = min(radius, max(width, height))

    x, y = np.meshgrid(np.arange(width), np.arange(height))
    dist_from_center = np.max(np.sqrt(((x - center_x)**2 + (y - center_y)**2)) - radius, 0)

    norm = dist_from_center / (max(center_x, center_y))
    norm = np.clip(norm, 0, 1)

    vignette_img = src_img * (1 - norm[..., np.newaxis])
    
    return vignette_img.astype(np.uint8)

def select_region(event, x, y, flags, param):
    src_img, region, pixel_size = param
    if event == cv.EVENT_LBUTTONDOWN or event == cv.EVENT_LBUTTONUP:
        region.append((x, y))
    if len(region) == 2:
        if  region[0] != region[1]:
            x0 = min(region[0][0],  region[1][0])
            y0 = min(region[0][1],  region[1][1])
            x1 = max(region[0][0],  region[1][0])
            y1 = max(region[0][1],  region[1][1])
            src_img = pixelate_region(src_img, x0, y0, x1, y1, pixel_size)
        region.clear()

def pixelate_region(src_img, x0, y0, x1, y1, numOfPixels):
    region = src_img[y0:y1, x0:x1]
    block_y = (y1-y0) // numOfPixels
    block_x = (x1-x0) // numOfPixels
    for i in range(block_y):
        for j in range(block_x):
            block = region[i*numOfPixels:(i+1)*numOfPixels, j*numOfPixels:(j+1)*numOfPixels]
            avg = np.mean(block, axis=(0, 1))
            region[i*numOfPixels:(i+1)*numOfPixels, j*numOfPixels:(j+1)*numOfPixels] = avg.astype(np.uint8)
    src_img[y0:y1, x0:x1] = region
    #tmp = resize(region, numOfPixels, numOfPixels)
    #pixelatedRegion = resize(tmp, x1-x0, y1-y0)
    #src_img[y0:y1, x0:x1] = pixelatedRegion
    return src_img