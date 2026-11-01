import numpy as np
import cv2 as cv
import os
from tkinter import Tk, filedialog
def resize_image(img): #Изменение размера
    height, width = img.shape[:2]
    try:
        new_width = int(input("Введите новую ширину: ")) #Новая ширина
        new_height = int(input("Введите новую высоту: ")) #Новая высота
        if new_height <= 0 or new_width <=0:
            print("")
            return
    except ValueError:
        print("")
        return

    resized_image = cv.resize(img, (new_width, new_height), interpolation=cv.INTER_AREA) #Изменение размеров
    return resized_image
def gray(img): #Перекрашивание фотографии в серый цвет
    b = img[:, :, 0].astype(np.float32) #Преобразование синего цвета в формат числа с плавающей запятой
    g = img[:, :, 1].astype(np.float32) #Преобразование зеленого цвета в формат числа с плавающей запятой
    r = img[:, :, 2].astype(np.float32) #Преобразование красного цвета в формат числа с плавающей запятой
    res = 0.114 * b + 0.587 * g + 0.299 * r #Перевод в отенки серого
    return res.astype(np.uint8) #Перевот ф формат целого числа
def sepia(img): #Эффект сепии
    kernel=np.array([
        [0.131, 0.534, 0.272],
        [0.168, 0.686, 0.349],
        [0.168, 0.769, 0.393]
    ])
    return cv.transform(img, kernel)
def vignette(img): #Виньетка
    h, w = img.shape[:2]
    mask = np.zeros((h,w), dtype=np.uint8)#Создаем черное изображения
    cv.circle(mask, (w//2, h//2), min(h,w), 255,-1)#Рисуем белый круг по центру
    mask = cv.GaussianBlur(mask, (101,101), 0)#Размытие
    mask = mask.astype(np.float32)/255 #Переводим значение маски в диапазон 0-1
    res = img.astype(np.float32)
    res *= mask[:, :, np.newaxis] #Умножаем изображение на маску
    return np.clip(res, 0, 255).astype(np.uint8) #Обрабатывает изображение
def antique(img): #Фильтр Антиквариат
    res=cv.pyrDown(img)
    res=cv.pyrUp(res) #Слегка размываем изображение
    res=sepia(res)
    res=vignette(res)
    return res
def fade(img, strenght = 0.5): #Фильтр выцветания
    white = np.ones(img.shape, dtype=np.uint8)* 255 #Создаем белый фон
    res = cv.addWeighted(img, 1-strenght, white, strenght, 0) #Смешиваем 2 изображения
    return res
def infrared(img):
    kernel = np.array([
        [0.4, 0.6, 0.0], #Новый B
        [0.3, -0.2, 0.9], #Новый G
        [-0.5, 1.4, 0.1] #Новый R
    ])
    res = cv.transform(img, kernel) #Пересчёт цветов
    return cv.convertScaleAbs(res, alpha = 1.15, beta = -10) #Увеличение контраста
def negative(img): #Негативный фильтр - инвертирование цветов
    return 255 - img
def matte(img):
    h, w = img.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)  # Создаем черное изображения
    cv.ellipse(mask, (w // 2, h // 2), (w // 2, h // 2), 0, 0, 360, 255, -1)  # Рисуем белый круг по центру
    mask = cv.GaussianBlur(mask, (101, 101), 0)  # Размытие
    mask = mask.astype(np.float32) / 255
    white = np.ones_like(img, dtype=np.float32)*255 #Создаем белый фон
    res = img.astype(np.float32)* mask[:, :, np.newaxis] + white * (1-mask[:, :, np.newaxis]) #Смешиваем овал и фото
    return np.clip(res, 0, 255).astype(np.uint8)
def old(img): #Эффект состаривания
    h, w= img.shape[:2]
    res = cv.convertScaleAbs(img, alpha = 0.8, beta = 20)
    #Добавляем шум
    noise = np.random.normal(0, 15, (h,w,3)) #оздает случайный шум
    res = res.astype(np.float32) + noise
    res = np.clip(res, 0, 255).astype(np.uint8)

    #Добавляем царапины
    for i in range(30):
        #Выбор случайных координатов
        x=np.random.randint(0,w)
        y=np.random.randint(0,h)
        #Цвет и длина царапины
        color = np.random.randint(150, 230)
        a=np.random.randint(10, h//4 +1)
        #Рисовка царапины
        cv.line(res,(x,y), (x+np.random.randint(-10,11), min(y+a,h-1)),(color,color,color),np.random.randint(1,3))
    return res

def neon(img, B=255, G=0, R=255, threshold=100):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY) #Перевод в серое изображение
    gray = cv.GaussianBlur(gray, (9, 9), 0) #Подавление шума

    lap = cv.Laplacian(gray, cv.CV_64F,  ksize=5) #Находим контуры оператором Лапласа
    lap = cv.convertScaleAbs(lap)
    mask = cv.threshold(lap, threshold, 255, cv.THRESH_BINARY)[1] #Создание маски контура

    #Раскрашиваем контуры
    neon = cv.merge([
        (mask * (B / 255)).astype(np.uint8),
        (mask * (G / 255)).astype(np.uint8),
        (mask * (R / 255)).astype(np.uint8),
    ])

    glow = cv.GaussianBlur(neon, (51, 51), 20) #Объединение контуров и размытого свечения
    neon = cv.add(neon, glow)

    #Тёмный фон + неон
    img_dark = cv.convertScaleAbs(img, alpha=0.6)
    return cv.add(img_dark, neon)


def choose_image(): #Выбор фотографии из папки Images
    folder = "Images"
    files = os.listdir(folder)
    images = []
    for file in files:
        if file.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
            images.append(file)
    if len(images) == 0:
        print("В папке нет фотографий!")
        return None
    print("Выберите фотографию:")
    for i in range(len(images)):
        print(i + 1, "-", images[i])
    choice = int(input("Введите номер фотографии: "))
    if choice < 1 or choice > len(images):
        print("Неверный номер!")
        return None
    path = os.path.join(folder, images[choice - 1])
    img = cv.imread(path)
    return img

def choose_filter(): #Выбор фильтра
    filters = {
        1: ("Resize", resize_image),
        2: ("Gray", gray),
        3: ("Antique", antique),
        4: ("Fade", fade),
        5: ("Infrared", infrared),
        6: ("Negative", negative),
        7: ("Matte", matte),
        8: ("Old", old),
        9: ("Neon", neon)
    }
    print("\nВыберите фильтр:")
    for number, (name, function) in filters.items():
        print(number, "-", name)
    choice = input("Введите номер фильтра: ")
    if not choice.isdigit():
        print("Введите число!")
        return None

    choice = int(choice)
    if choice not in filters:
        print("Такого фильтра нет!")
        return None
    return filters[choice]