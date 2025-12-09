import cv2
import numpy as np
import sys

def pixelize_region(image, x, y, width, height, pixel_size=10):
    img = image.copy()
    x2, y2 = x + width, y + height
    for i in range(y, min(y2, img.shape[0]), pixel_size):
        for j in range(x, min(x2, img.shape[1]), pixel_size):
            h_end = min(i + pixel_size, y2, img.shape[0])
            w_end = min(j + pixel_size, x2, img.shape[1])
            block = img[i:h_end, j:w_end]
            if block.size > 0:
                avg = np.mean(block, axis=(0, 1)).astype(np.uint8)
                img[i:h_end, j:w_end] = avg
    return img

class Selector:
    def __init__(self, image):
        self.img = image
        self.display = image.copy()
        self.start = (-1, -1)
        self.end = (-1, -1)
        self.drawing = False
        self.result = None

    def mouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.start = (x, y)
        elif event == cv2.EVENT_MOUSEMOVE and self.drawing:
            self.display = self.img.copy()
            cv2.rectangle(self.display, self.start, (x, y), (0, 255, 0), 2)
        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            self.end = (x, y)
            x1, y1 = min(self.start[0], x), min(self.start[1], y)
            x2, y2 = max(self.start[0], x), max(self.start[1], y)
            if x2 > x1 and y2 > y1:
                self.result = pixelize_region(self.img, x1, y1, x2-x1, y2-y1, 20)
                cv2.imshow("Результат", self.result)

    def run(self):
        cv2.namedWindow("Выделите область")
        cv2.setMouseCallback("Выделите область", self.mouse)
        print("Зажмите ЛКМ, выделите область, отпустите. Нажмите 'q' для выхода.")
        while True:
            cv2.imshow("Выделите область", self.display)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cv2.destroyAllWindows()

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Использование: python pixelize_tool.py фото.jpg")
        sys.exit()
    
    image = cv2.imread(sys.argv[1])
    if image is None:
        print("Ошибка: изображение не найдено")
        sys.exit()
        
    Selector(image).run()