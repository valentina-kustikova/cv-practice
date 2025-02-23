import cv2
import numpy as np
import argparse


# Организация работы с аргументами командной строки
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--image_path', type = str)
    parser.add_argument('-f', '--filter', type = str,
                        choices = ['grayscale', 'resize', 'sepia',
                                   'vignette', 'pixelate'])

    parser.add_argument('-r', '--resizeValue', type = float, default = 1)
    parser.add_argument('-v', '--vignetteRadius', type = float, default = 500)
    parser.add_argument('-p', '--pixelationBlockSize', type = int, default = 10)

    return parser.parse_args()


# Функция для вывода изображения на экран
def picToScreen(name: str,
                image: np.ndarray) -> None:
    cv2.imshow(name, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# Определение фильтра grayscale
def grayscale(image: np.ndarray) -> np.ndarray:
    # Проверка корректности изображения
    if image.shape[2] != 3:
        raise RuntimeError("Image must have 3 channels or it was opened with an error.")

    # Разделяем изображение на три канала
    B, G, R = cv2.split(image)

    # Вычисляем фильтр
    grayImage = (R * 0.2989 + G * 0.5870 + B * 0.1140).astype(np.uint8)
    
    return grayImage


# Определение функции, изменяющей размер изображения
def resize(image: np.ndarray,
           value: float) -> np.ndarray:
    # Проверка корректности значения value
    if value < 0:
        raise RuntimeError('Value must be greater than zero')

    # Получаем размеры исходного изображения
    height, width = image.shape[0], image.shape[1]

    # Изменяем размеры изображения в value раз
    newWidth = int(width * value)
    newHeight = int(height * value)

    # Вычисляем координаты пикселей, которые будут использоваться в итоговом изображении
    x = np.floor(np.arange(newWidth) / value).astype(int)
    y = np.floor(np.arange(newHeight) / value).astype(int)

    # Делаем из y вектор-столбец и собираем итоговую матрицу
    resizedImage = image[y[:, None], x]
    return resizedImage


# Определение функции, дающей изображению фотоэффект сепии
def sepia(image: np.ndarray) -> np.ndarray:
    # Проверка корректности изображения
    if image.shape[2] != 3:
        raise RuntimeError("Image must have 3 channels or it was opened with an error.")

    # Разделяем изображение на три канала
    B, G, R = cv2.split(image)

    # Применяем фильтр к каждому каналу
    newR = ((0.393 * R) + (0.769 * G) + (0.189 * B)).clip(0, 255).astype(np.uint8)
    newG = ((0.349 * R) + (0.686 * G) + (0.168 * B)).clip(0, 255).astype(np.uint8)
    newB = ((0.272 * R) + (0.534 * G) + (0.131 * B)).clip(0, 255).astype(np.uint8)

    # Собираем все вместе
    sepiaImage = cv2.merge([newB, newG, newR])
    return sepiaImage


# Определение функции, дающей изображению фотоэффект виньетки
def vignette(image: np.ndarray,
             radius: float) -> np.ndarray:
    # Проверка корректности радиуса
    if radius <= 0:
        raise RuntimeError("Radius must be greater then zero.")

    # Получаем размеры исходного изображения
    height, width = image.shape[0], image.shape[1]

    # Определяем центр изображения
    y, x = height // 2, width // 2

    # Создаем два массива:
    # yInd представляет собой матрицу, которая вместо каждого элемента содержит индекс текущей строки
    # xInd - содержит вместо элементов индексы столбцов
    yInd, xInd = np.indices((height, width))

    # Вычисляем расстояние от каждого элемента до центра матрицы
    distances = np.sqrt((xInd - x) ** 2 + (yInd - y) ** 2)

    # Создаем маску, которая представляет собой матрицу расстояний между конкретной ячейкой и центром матрицы,
    # при этом все расстояния нормируются до [0, 1]
    mask = np.clip(1 - distances / radius, 0, 1)

    # Умножаем значение каждого цветового канала на соответствующее значение маски и получаем ответ
    B, G, R = cv2.split(image)
    B = (B * mask).astype(np.uint8)
    G = (G * mask).astype(np.uint8)
    R = (R * mask).astype(np.uint8)

    vignetteImage = cv2.merge([B, G, R])
    return vignetteImage


# Переменные, которые нужно определить глобально для корректной работы функции пикселизации
x1, y1, x2, y2 = -1, -1, -1, -1
drawing = False


# Определение пикселизации заданной прямоугольной области изображения
def pixel_filter(image: np.ndarray,
                 size: int) -> np.ndarray:
    global x1, y1, x2, y2

    # Если x1 == y1 == x2 == y2 == -1, то это означает, что пользователь не выбрал область и
    # пикселизироваться будет все изображение
    if x1 == y1 == x2 == y2 == -1:
        x1, y1 = 0, 0
        x2, y2 = image.shape[1], image.shape[0]

    # Выбираем нужную область, а также её размер
    block = image[y1:y2, x1:x2]
    blockHeight, blockWidth = block.shape[0], block.shape[1]

    # Нормируем размер относительно параметра
    height = blockHeight // size
    width = blockWidth // size

    for i in range(height):
        # Вычисляем индексы начала и конца обрабатываемого блока
        startY = i * size
        endY = min(startY + size, blockHeight)
        for j in range(width):
            startX = j * size
            endX = min(startX + size, blockWidth)

            block[startY:endY, startX:endX] = (block[startY:endY, startX:endX].mean(axis = (0, 1))).astype(int)

    image[y1:y2, x1:x2] = block
    return image


# Функция, отвечающая за обработку события мыши для рисования прямоугольника и пикселизации выбранной области
def draw(event: int,
         x: int, y: int,
         flags: int, param: None) -> None:
    global drawing, x1, y1, x2, y2

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        x1, y1 = x, y

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            x2, y2 = x, y

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        x2, y2 = x, y


# Сама функция пикселизации
def pixelImage(image: np.ndarray,
                size: int) -> np.ndarray:
    # Проверка корректности размера блока
    if size <= 0:
        raise RuntimeError("Block size must be greater then zero.")

    global drawing, x1, y1, x2, y2

    cv2.namedWindow('Image without pixelation')
    cv2.setMouseCallback('Image without pixelation', draw)

    while True:
        img_copy = image.copy()

        # Отображение выбранной прямоугольной области на экран
        if x1 != -1 and y1 != -1 and x2 != -1 and y2 != -1:
            cv2.rectangle(img_copy, (x1, y1), (x2, y2), (0, 255, 0), 2)

        cv2.imshow('Image without pixelation', img_copy)

        # Если нажата нужная кнопка (p), то появляется пикселизированное изображение
        key = cv2.waitKey(1)
        if key == ord('p'):
            cv2.destroyWindow('Image without pixelation')
            break
    return pixel_filter(image, size)


def main():
    args = parse_args()
    image_path = args.image_path
    image = cv2.imread(image_path)

    picToScreen('Original image', image)

    if args.filter == 'grayscale':
        grayImage = grayscale(image)
        picToScreen('Gray image', grayImage)
    elif args.filter == 'resize':
        resizedImage = resize(image, args.resizeValue)
        picToScreen('Resized image', resizedImage)
    elif args.filter == 'sepia':
        sepiaImage = sepia(image)
        picToScreen('Sepia image', sepiaImage)
    elif args.filter == 'vignette':
        vignetteImage = vignette(image, args.vignetteRadius)
        picToScreen('Vignette image', vignetteImage)
    elif args.filter == 'pixelate':
        pixelatedImage = pixelImage(image, args.pixelationBlockSize)
        picToScreen('Pixelated image', pixelatedImage)


if __name__ == '__main__':
    main()
