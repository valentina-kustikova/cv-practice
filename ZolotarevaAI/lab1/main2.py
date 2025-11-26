import cv2
import numpy as np
from filters import pixelate_region

drawing = False #фиксируем удержание лкм
ix, iy = -1, -1
image = cv2.imread("example.jpg")
orig = image.copy()

def draw_rectangle(event, x, y, flags, param):
    global ix, iy, drawing, image
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y #начальные координаты

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            temp = image.copy()
            cv2.rectangle(temp, (ix, iy), (x, y), (0, 255, 0), 1)
            cv2.imshow("Image", temp)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        w, h = abs(x - ix), abs(y - iy)
        x1, y1 = min(ix, x), min(iy, y)
        image = pixelate_region(image, x1, y1, w, h, pixel_size=15)
        cv2.imshow("Image", image)

cv2.imshow("Image", image)
cv2.setMouseCallback("Image", draw_rectangle) #обработчик событий мышки

cv2.waitKey(0)
cv2.destroyAllWindows()
